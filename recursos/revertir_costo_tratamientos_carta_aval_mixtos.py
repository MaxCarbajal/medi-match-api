# Revierte a precio normal los pares (proveedor, tratamiento) que
# actualizar_costo_tratamientos_carta_aval.py (2026-09-16) dejó con el precio
# CARTA AVAL, pero que TAMBIÉN tienen filas MEDICINA PREVENTIVA reales -- ver
# docs/DECISIONES.md 2026-09-16 ("ahorro máximo"). De los 110 pares con
# alguna fila CARTA AVAL en Datos_tratramiento_detalle_final.xlsx, 16 tienen
# AMBOS tipos de fila; ese script les puso el precio de trámite de reembolso
# (mucho más alto) como coste_medio, aunque el proveedor sí ofrece el
# tratamiento a precio normal. Pedido explícito del usuario: revertir esos a
# su precio MEDICINA PREVENTIVA. Los otros 94 pares (SOLO CARTA AVAL, sin
# precio normal) NO se tocan -- ese es el precio real que tienen.
#
# coste_medio = promedio de SUMA entre las filas MEDICINA PREVENTIVA del par
# (mismo criterio de agregación que los otros scripts de esta carpeta).
#
# Usa la service_role key de medi-match-api/.env (no el conector MCP de
# Supabase de esta sesión, que no ve este proyecto -- ver CLAUDE.md).
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

RUTA_EXCEL = "../../medi-match-db/dataset/Datos_tratramiento_detalle_final.xlsx"

load_dotenv("../../medi-match-api/.env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

print("Leyendo Excel...")
df = pd.read_excel(RUTA_EXCEL)
df.columns = [c.strip() for c in df.columns]
df["CODPROVEEDOR"] = df["CODPROVEEDOR"].astype(int)
df["CODIGO_TRATAMIENTO"] = df["CODIGO_TRATAMIENTO"].astype(int)

tipos = df.groupby(["CODPROVEEDOR", "CODIGO_TRATAMIENTO"])["DESCRIPTIPOPRE"].apply(lambda s: set(s))
pares_mixtos = tipos[tipos.apply(lambda s: len(s) > 1)].index
print(f"  pares con ambos tipos (MEDICINA PREVENTIVA + CARTA AVAL): {len(pares_mixtos)}")

mp = df[df["DESCRIPTIPOPRE"] == "MEDICINA PREVENTIVA"]
nuevo = (
    mp.groupby(["CODPROVEEDOR", "CODIGO_TRATAMIENTO"])
    .agg(coste_medio=("SUMA", "mean"), tratamiento=("DESCPROCED", "first"), id_municipio=("CODIGO_MUNICIPIO", "first"), pasar_modelo=("pasar_modelo", "first"))
    .reset_index()
    .rename(columns={"CODPROVEEDOR": "id_proveedor", "CODIGO_TRATAMIENTO": "id_tratamiento"})
)

filas_completas = {}
for f in nuevo.itertuples(index=False):
    clave = (f.id_proveedor, f.id_tratamiento)
    if clave not in pares_mixtos:
        continue
    filas_completas[clave] = {
        "id_proveedor": f.id_proveedor,
        "id_tratamiento": f.id_tratamiento,
        "tratamiento": f.tratamiento,
        "id_municipio": f.id_municipio,
        "pasar_modelo": bool(f.pasar_modelo),
        "coste_medio": round(float(f.coste_medio), 2),
    }

print(f"  pares mixtos con precio MEDICINA PREVENTIVA calculable: {len(filas_completas)}")

print("\nDescargando coste_medio actual de esos pares en Supabase...")
actuales = {}
for id_proveedor, id_tratamiento in filas_completas:
    resp = (
        supabase.table("costo_tratamientos")
        .select("id_proveedor, id_tratamiento, coste_medio")
        .eq("id_proveedor", id_proveedor)
        .eq("id_tratamiento", id_tratamiento)
        .execute()
        .data
    )
    if resp:
        actuales[(id_proveedor, id_tratamiento)] = float(resp[0]["coste_medio"])

filas_a_actualizar = []
print("\nComparación (actual en Supabase, hoy con precio CARTA AVAL) vs precio MEDICINA PREVENTIVA real:")
for clave, fila in filas_completas.items():
    actual = actuales.get(clave)
    print(f"  proveedor={clave[0]} tratamiento={clave[1]} ({fila['tratamiento'][:50]}) -- actual={actual} -> nuevo(MEDICINA PREVENTIVA)={fila['coste_medio']}")
    if actual is None or abs(actual - fila["coste_medio"]) > 0.0001:
        filas_a_actualizar.append(fila)

print(f"\nPares a revertir: {len(filas_a_actualizar):,} / {len(filas_completas):,}")
if not filas_a_actualizar:
    print("Nada que actualizar.")
    sys.exit(0)

confirmar = input("\n¿Revertir estos pares a su precio MEDICINA PREVENTIVA en Supabase? (escribir 'si' para continuar): ")
if confirmar.strip().lower() != "si":
    print("Cancelado -- no se cambió nada.")
    sys.exit(0)

for fila in filas_a_actualizar:
    supabase.table("costo_tratamientos").upsert(fila, on_conflict="id_proveedor,id_tratamiento").execute()
    print(f"  actualizado: proveedor={fila['id_proveedor']} tratamiento={fila['id_tratamiento']} -> coste_medio={fila['coste_medio']}")

print("\nListo.")
