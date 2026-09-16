# Actualiza costo_tratamientos.coste_medio con el precio nuevo de
# medi-match-db/dataset/Datos_tratramiento_detalle.xlsx (2026-09-15) -- el
# usuario confirmó que es el mismo dataset original con el precio (columna
# SUMA) actualizado, no un rediseño: mismas 9.261 combinaciones
# (CODPROVEEDOR, CODIGO_TRATAMIENTO), mismo pasar_modelo, mismo
# CODIGO_MUNICIPIO, mismo rating_global -- solo cambia SUMA en 8.065/9.261
# pares (verificado por comparación 1:1 contra
# datos_tratamientos_detalle_con_rating_limpio.xlsx, el dataset ya cargado).
#
# Solo escritura sobre coste_medio en costo_tratamientos -- no toca
# proveedores, capacidades ni asignaciones. coste_medio se recalcula como el
# promedio de SUMA por (CODPROVEEDOR, CODIGO_TRATAMIENTO), igual que en el
# dataset original (72.391 líneas de facturación -> 9.261 pares).
#
# El diff se hace contra el estado REAL de Supabase (no contra el Excel
# viejo local, que podría haber quedado desactualizado) -- se descarga
# costo_tratamientos completo, se compara, y solo se suben los pares cuyo
# coste_medio cambió. Pares del Excel que no existan en la base (no debería
# haber ninguno) se reportan y NO se insertan acá -- insertar necesitaría
# tratamiento/id_municipio, que este script no toca.
#
# Usa la service_role key de medi-match-api/.env (no el conector MCP de
# Supabase de esta sesión, que no ve este proyecto -- ver CLAUDE.md).
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

RUTA_EXCEL = "../../medi-match-db/dataset/Datos_tratramiento_detalle.xlsx"
TAMANO_LOTE = 500

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

nuevo = (
    df.groupby(["CODPROVEEDOR", "CODIGO_TRATAMIENTO"])
    .agg(coste_medio=("SUMA", "mean"), tratamiento=("DESCPROCED", "first"), id_municipio=("CODIGO_MUNICIPIO", "first"), pasar_modelo=("pasar_modelo", "first"))
    .reset_index()
    .rename(columns={"CODPROVEEDOR": "id_proveedor", "CODIGO_TRATAMIENTO": "id_tratamiento"})
)
# Fila completa por par -- necesaria para el upsert: on conflict, PostgREST solo
# preserva las columnas que sí se envían en el payload, así que si se manda
# nada más que id_proveedor/id_tratamiento/coste_medio, el resto (tratamiento,
# id_municipio -- NOT NULL) se pisa con NULL y el upsert falla (verificado).
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
}
pares_excel = {clave: fila["coste_medio"] for clave, fila in filas_completas.items()}

print("Descargando costo_tratamientos actual de Supabase...")
pares_actuales = descargar_costo_tratamientos_actual()
print(f"  filas en la base: {len(pares_actuales):,}")
print(f"  pares en el Excel: {len(pares_excel):,}")

solo_en_excel = [p for p in pares_excel if p not in pares_actuales]
solo_en_base = [p for p in pares_actuales if p not in pares_excel]

if solo_en_excel:
    print(f"\nAVISO: {len(solo_en_excel)} pares del Excel no existen en la base -- NO se insertan (requeriría tratamiento/id_municipio). Ejemplos: {solo_en_excel[:10]}")
if solo_en_base:
    print(f"\nAVISO: {len(solo_en_base)} pares de la base no vinieron en el Excel -- se dejan sin tocar. Ejemplos: {solo_en_base[:10]}")

filas_a_actualizar = []
for clave, coste_nuevo in pares_excel.items():
    if clave not in pares_actuales:
        continue
    coste_actual = round(pares_actuales[clave], 2)
    if abs(coste_nuevo - coste_actual) > 0.0001:
        filas_a_actualizar.append(filas_completas[clave])

print(f"\nPares con coste_medio distinto: {len(filas_a_actualizar):,} / {len(pares_excel):,}")
if filas_a_actualizar:
    ejemplo = filas_a_actualizar[:5]
    print("Ejemplos:", ejemplo)

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
