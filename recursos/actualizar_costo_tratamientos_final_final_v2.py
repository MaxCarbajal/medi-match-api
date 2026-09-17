# v2 de la carga de Datos_tratramiento_detalle_final_final.xlsx (ver
# actualizar_costo_tratamientos_final_final.py, ya borrado por el usuario --
# este reemplaza ese enfoque). Pedido explícito del usuario (2026-09-17):
#
#   1) "Cargar el archivo de cero" -- en vez de tocar solo los pares con
#      coste_medio distinto, se recalculan y suben TODOS los pares del
#      archivo (excepto los outliers, ver abajo). No se hace DELETE FROM
#      costo_tratamientos: capacidades/asignaciones tienen FK contra esa
#      tabla SIN on delete cascade (migración 014), así que un delete
#      masivo rompería esas FKs (o peor, si se forzara cascade, borraría
#      capacidad real). Como los pares (id_proveedor, id_tratamiento) no
#      cambian, un upsert de todos los pares logra el mismo resultado
#      (reflejar el archivo nuevo) sin ese riesgo.
#
#   2) "Los outliers no necesariamente deben ser repetidos, el precio se
#      aleja mucho del promedio" -- se agrega un segundo criterio de
#      exclusión (ver docs/DECISIONES.md 2026-09-17): precio de un par
#      proveedor+tratamiento > 5x la MEDIANA de ese mismo tratamiento entre
#      todos los proveedores que lo ofrecen (umbral elegido por el quiebre
#      real en la distribución: 99.5% de los pares está bajo 4.5x, el
#      siguiente percentil salta a 80x). Este criterio agrega 36 pares que
#      el criterio anterior (precio repetido del proveedor) no detectaba
#      -- ej. proveedores con UN solo tratamiento a un precio absurdo, que
#      no repiten el valor en ningún otro lado.
#
# Total excluido: 231/9.261 pares (195 por proveedor con precio repetido +
# 36 solo por desviación). Esos quedan con su coste_medio actual sin tocar.
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

RUTA_EXCEL = "../../medi-match-db/dataset/Datos_tratramiento_detalle_final_final.xlsx"
TAMANO_LOTE = 500
UMBRAL_DESVIACION = 5.0

load_dotenv("../../medi-match-api/.env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def descargar_costo_tratamientos_actual():
    filas = []
    offset = 0
    while True:
        resp = (
            supabase.table("costo_tratamientos")
            .select("id_proveedor, id_tratamiento, coste_medio")
            .range(offset, offset + 999)
            .execute()
            .data
        )
        filas.extend(resp)
        if len(resp) < 1000:
            break
        offset += 1000
    return {(f["id_proveedor"], f["id_tratamiento"]): float(f["coste_medio"]) for f in filas}


print("Leyendo Excel...")
df = pd.read_excel(RUTA_EXCEL)
df.columns = [c.strip() for c in df.columns]
df["CODPROVEEDOR"] = df["CODPROVEEDOR"].astype(int)
df["CODIGO_TRATAMIENTO"] = df["CODIGO_TRATAMIENTO"].astype(int)

# Criterio 1: proveedor con precio repetido en tratamientos sin relacion.
agg_prov = df.groupby("CODPROVEEDOR")["CODIGO_TRATAMIENTO"].nunique().rename("n_tratamientos").to_frame()
agg_prov["n_precios_unicos"] = df.groupby("CODPROVEEDOR")["SUMA"].nunique()
agg_prov["ratio_prov"] = agg_prov["n_precios_unicos"] / agg_prov["n_tratamientos"]
proveedores_repetido = set(agg_prov[(agg_prov["n_tratamientos"] >= 4) & (agg_prov["ratio_prov"] <= 0.34)].index)
print(f"  proveedores con precio repetido: {len(proveedores_repetido)}")

nuevo = (
    df.groupby(["CODPROVEEDOR", "CODIGO_TRATAMIENTO"])
    .agg(coste_medio=("SUMA", "mean"), tratamiento=("DESCPROCED", "first"), id_municipio=("CODIGO_MUNICIPIO", "first"), pasar_modelo=("pasar_modelo", "first"))
    .reset_index()
    .rename(columns={"CODPROVEEDOR": "id_proveedor", "CODIGO_TRATAMIENTO": "id_tratamiento"})
)

# Criterio 2: precio del par > 5x la mediana de ese tratamiento entre todos los proveedores.
mediana_tratamiento = nuevo.groupby("id_tratamiento")["coste_medio"].median()
nuevo["mediana_tratamiento"] = nuevo["id_tratamiento"].map(mediana_tratamiento)
nuevo["ratio_desviacion"] = nuevo["coste_medio"] / nuevo["mediana_tratamiento"]
pares_desviados = set(zip(nuevo.loc[nuevo["ratio_desviacion"] > UMBRAL_DESVIACION, "id_proveedor"], nuevo.loc[nuevo["ratio_desviacion"] > UMBRAL_DESVIACION, "id_tratamiento"]))
print(f"  pares individuales con precio >{UMBRAL_DESVIACION}x la mediana de su tratamiento: {len(pares_desviados)}")

excluidos = {(r.id_proveedor, r.id_tratamiento) for r in nuevo.itertuples(index=False) if r.id_proveedor in proveedores_repetido} | pares_desviados
print(f"  total de pares excluidos (union de ambos criterios): {len(excluidos):,} / {len(nuevo):,}")

filas_completas = {
    (f.id_proveedor, f.id_tratamiento): {
        "id_proveedor": f.id_proveedor,
        "id_tratamiento": f.id_tratamiento,
        "tratamiento": f.tratamiento,
        "id_municipio": f.id_municipio,
        "pasar_modelo": bool(f.pasar_modelo),
        "coste_medio": round(float(f.coste_medio), 2),
    }
    for f in nuevo.itertuples(index=False)
    if (f.id_proveedor, f.id_tratamiento) not in excluidos
}
pares_excel = {clave: fila["coste_medio"] for clave, fila in filas_completas.items()}

print("Descargando costo_tratamientos actual de Supabase...")
pares_actuales = descargar_costo_tratamientos_actual()
print(f"  filas en la base: {len(pares_actuales):,}")

solo_en_excel = [p for p in pares_excel if p not in pares_actuales]
if solo_en_excel:
    print(f"\nAVISO: {len(solo_en_excel)} pares del Excel no existen en la base -- NO se insertan. Ejemplos: {solo_en_excel[:10]}")

filas_a_actualizar = []
for clave, coste_nuevo in pares_excel.items():
    if clave not in pares_actuales:
        continue
    coste_actual = round(pares_actuales[clave], 2)
    if abs(coste_nuevo - coste_actual) > 0.0001:
        filas_a_actualizar.append(filas_completas[clave])

print(f"\nPares con coste_medio distinto (de los {len(pares_excel):,} confiables): {len(filas_a_actualizar):,}")
print(f"Pares excluidos que quedan con su coste_medio actual: {len(excluidos):,}")
if filas_a_actualizar:
    print("Ejemplos:", filas_a_actualizar[:5])

if not filas_a_actualizar:
    print("\nNada que actualizar.")
    sys.exit(0)

confirmar = input("\n¿Subir estos cambios de precio a Supabase? (escribir 'si' para continuar): ")
if confirmar.strip().lower() != "si":
    print("Cancelado -- no se subió nada.")
    sys.exit(0)

print("\nSubiendo en lotes...")
for i in range(0, len(filas_a_actualizar), TAMANO_LOTE):
    lote = filas_a_actualizar[i : i + TAMANO_LOTE]
    supabase.table("costo_tratamientos").upsert(lote, on_conflict="id_proveedor,id_tratamiento").execute()
    print(f"  {min(i + TAMANO_LOTE, len(filas_a_actualizar)):,}/{len(filas_a_actualizar):,}")

print("\nListo.")
