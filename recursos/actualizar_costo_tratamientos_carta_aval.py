# Actualiza costo_tratamientos.coste_medio SOLO para los pares
# (proveedor, tratamiento) que tienen filas DESCRIPTIPOPRE == 'CARTA AVAL' en
# medi-match-db/dataset/Datos_tratramiento_detalle_final.xlsx (2026-09-16).
#
# Pedido explícito del usuario: usar ÚNICAMENTE las filas marcadas CARTA AVAL
# para calcular el precio de esos pares -- se descarta el resto del archivo
# (las 71.882 filas MEDICINA PREVENTIVA) para este cálculo. De los 9.261
# pares del dataset, solo 110 tienen alguna fila CARTA AVAL (509 filas en
# total) -- son los únicos que este script toca. El resto de
# costo_tratamientos (9.151 pares) queda intacto.
#
# coste_medio = promedio de SUMA entre las filas CARTA AVAL de cada par
# (mismo criterio de agregación que actualizar_costo_tratamientos.py).
#
# Fila completa en el upsert (tratamiento/id_municipio/pasar_modelo,
# NOT NULL sin default) -- ver nota en actualizar_costo_tratamientos.py: un
# upsert con menos columnas pisa el resto con NULL y falla.
#
# Usa la service_role key de medi-match-api/.env (no el conector MCP de
# Supabase de esta sesión, que no ve este proyecto -- ver CLAUDE.md).
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

RUTA_EXCEL = "../../medi-match-db/dataset/Datos_tratramiento_detalle_final.xlsx"
TAMANO_LOTE = 500

load_dotenv("../../medi-match-api/.env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def descargar_costo_tratamientos_actual(pares):
    """Descarga solo las filas de la base que corresponden a los pares pedidos."""
    filas = {}
    proveedores = sorted({p[0] for p in pares})
    for id_proveedor in proveedores:
        resp = (
            supabase.table("costo_tratamientos")
            .select("id_proveedor, id_tratamiento, coste_medio")
            .eq("id_proveedor", id_proveedor)
            .execute()
            .data
        )
        for f in resp:
            filas[(f["id_proveedor"], f["id_tratamiento"])] = float(f["coste_medio"])
    return filas


print("Leyendo Excel...")
df = pd.read_excel(RUTA_EXCEL)
df.columns = [c.strip() for c in df.columns]
df["CODPROVEEDOR"] = df["CODPROVEEDOR"].astype(int)
df["CODIGO_TRATAMIENTO"] = df["CODIGO_TRATAMIENTO"].astype(int)

carta_aval = df[df["DESCRIPTIPOPRE"] == "CARTA AVAL"]
print(f"  filas totales: {len(df):,} | filas CARTA AVAL: {len(carta_aval):,}")

nuevo = (
    carta_aval.groupby(["CODPROVEEDOR", "CODIGO_TRATAMIENTO"])
    .agg(coste_medio=("SUMA", "mean"), tratamiento=("DESCPROCED", "first"), id_municipio=("CODIGO_MUNICIPIO", "first"), pasar_modelo=("pasar_modelo", "first"))
    .reset_index()
    .rename(columns={"CODPROVEEDOR": "id_proveedor", "CODIGO_TRATAMIENTO": "id_tratamiento"})
)
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
print(f"  pares únicos con CARTA AVAL: {len(pares_excel):,}")

print("Descargando de Supabase el estado actual de esos pares...")
pares_actuales = descargar_costo_tratamientos_actual(pares_excel.keys())
print(f"  pares encontrados en la base: {len(pares_actuales):,}")

solo_en_excel = [p for p in pares_excel if p not in pares_actuales]
if solo_en_excel:
    print(f"\nAVISO: {len(solo_en_excel)} pares CARTA AVAL no existen en la base -- NO se insertan. Ejemplos: {solo_en_excel[:10]}")

filas_a_actualizar = []
for clave, coste_nuevo in pares_excel.items():
    if clave not in pares_actuales:
        continue
    coste_actual = round(pares_actuales[clave], 2)
    if abs(coste_nuevo - coste_actual) > 0.0001:
        filas_a_actualizar.append(filas_completas[clave])

print(f"\nPares con coste_medio distinto: {len(filas_a_actualizar):,} / {len(pares_excel):,}")
if filas_a_actualizar:
    print("Todas las filas a actualizar:")
    for fila in filas_a_actualizar:
        print(f"  proveedor={fila['id_proveedor']} tratamiento={fila['id_tratamiento']} ({fila['tratamiento'][:60]}) -> coste_medio={fila['coste_medio']}")

if not filas_a_actualizar:
    print("\nNada que actualizar.")
    sys.exit(0)

confirmar = input("\n¿Subir estos cambios de precio (solo CARTA AVAL) a Supabase? (escribir 'si' para continuar): ")
if confirmar.strip().lower() != "si":
    print("Cancelado -- no se subió nada.")
    sys.exit(0)

print("\nSubiendo en lotes...")
for i in range(0, len(filas_a_actualizar), TAMANO_LOTE):
    lote = filas_a_actualizar[i : i + TAMANO_LOTE]
    supabase.table("costo_tratamientos").upsert(lote, on_conflict="id_proveedor,id_tratamiento").execute()
    print(f"  {min(i + TAMANO_LOTE, len(filas_a_actualizar)):,}/{len(filas_a_actualizar):,}")

print("\nListo.")
