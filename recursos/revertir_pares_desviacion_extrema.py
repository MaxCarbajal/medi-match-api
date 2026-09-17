# Revierte a un precio confiable los 36 pares que quedaron con un precio
# outlier en costo_tratamientos (subidos por
# actualizar_costo_tratamientos_final_final.py, 2026-09-17, antes de tener
# el criterio de desviación -- ver docs/DECISIONES.md 2026-09-17).
#
# Criterio de outlier (nuevo, pedido explícito del usuario: "no
# necesariamente deben ser repetidos, el precio se aleja mucho del
# promedio"): precio del par > 5x la mediana de ese mismo tratamiento entre
# todos los proveedores, en Datos_tratramiento_detalle_final_final.xlsx.
#
# Para la mayoría de estos 36 pares, ese mismo archivo solo tiene 1 fila (el
# valor outlier es el único dato ahí, no hay alternativa dentro del
# archivo) -- así que se usa una fuente distinta: el dataset ORIGINAL
# (datos_tratamientos_detalle_con_rating_limpio.xlsx, el primero cargado a
# este proyecto), que tiene valores sanos y cercanos a la mediana para los
# 36 pares (verificado 1:1, sin excepciones).
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

RUTA_FINAL_FINAL = "../../medi-match-db/dataset/Datos_tratramiento_detalle_final_final.xlsx"
RUTA_ORIGINAL = "../../medi-match-db/dataset/datos_tratamientos_detalle_con_rating_limpio.xlsx"
UMBRAL_DESVIACION = 5.0

load_dotenv("../../medi-match-api/.env")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

print("Leyendo Excel final_final...")
ff = pd.read_excel(RUTA_FINAL_FINAL)
ff.columns = [c.strip() for c in ff.columns]
ff["CODPROVEEDOR"] = ff["CODPROVEEDOR"].astype(int)
ff["CODIGO_TRATAMIENTO"] = ff["CODIGO_TRATAMIENTO"].astype(int)

agg_prov = ff.groupby("CODPROVEEDOR")["CODIGO_TRATAMIENTO"].nunique().rename("n_tratamientos").to_frame()
agg_prov["n_precios_unicos"] = ff.groupby("CODPROVEEDOR")["SUMA"].nunique()
agg_prov["ratio_prov"] = agg_prov["n_precios_unicos"] / agg_prov["n_tratamientos"]
proveedores_repetido = set(agg_prov[(agg_prov["n_tratamientos"] >= 4) & (agg_prov["ratio_prov"] <= 0.34)].index)

pares = ff.groupby(["CODPROVEEDOR", "CODIGO_TRATAMIENTO"]).agg(precio=("SUMA", "mean"), tratamiento=("DESCPROCED", "first"), id_municipio=("CODIGO_MUNICIPIO", "first"), pasar_modelo=("pasar_modelo", "first")).reset_index()
mediana_trat = pares.groupby("CODIGO_TRATAMIENTO")["precio"].median().rename("mediana_tratamiento")
pares = pares.merge(mediana_trat, on="CODIGO_TRATAMIENTO")
pares["ratio"] = pares["precio"] / pares["mediana_tratamiento"]

solo_desviacion = pares[(pares["ratio"] > UMBRAL_DESVIACION) & (~pares["CODPROVEEDOR"].isin(proveedores_repetido))]
print(f"  pares a revertir (solo desviacion, no proveedor repetido): {len(solo_desviacion)}")

print("Leyendo dataset original...")
orig = pd.read_excel(RUTA_ORIGINAL)
orig.columns = [c.strip() for c in orig.columns]
orig = orig.rename(columns={"CODIGO": "CODIGO_TRATAMIENTO"})
orig["CODPROVEEDOR"] = orig["CODPROVEEDOR"].astype(int)
orig["CODIGO_TRATAMIENTO"] = orig["CODIGO_TRATAMIENTO"].astype(int)
precio_original = orig.groupby(["CODPROVEEDOR", "CODIGO_TRATAMIENTO"])["SUMA"].mean().rename("precio_original").reset_index()

candidatos = solo_desviacion.merge(precio_original, on=["CODPROVEEDOR", "CODIGO_TRATAMIENTO"], how="left")
sin_original = candidatos[candidatos["precio_original"].isna()]
if len(sin_original):
    print(f"  AVISO: {len(sin_original)} pares sin valor en el dataset original -- NO se pueden revertir, se dejan como estan:")
    print(sin_original[["CODPROVEEDOR", "tratamiento"]].to_string(index=False))
candidatos = candidatos.dropna(subset=["precio_original"])

print("\nDescargando coste_medio actual de Supabase para estos pares...")
filas_a_actualizar = []
for f in candidatos.itertuples(index=False):
    resp = (
        supabase.table("costo_tratamientos")
        .select("id_proveedor, id_tratamiento, coste_medio")
        .eq("id_proveedor", f.CODPROVEEDOR)
        .eq("id_tratamiento", f.CODIGO_TRATAMIENTO)
        .execute()
        .data
    )
    if not resp:
        continue
    actual = float(resp[0]["coste_medio"])
    nuevo_precio = round(float(f.precio_original), 2)
    print(f"  proveedor={f.CODPROVEEDOR} tratamiento={f.CODIGO_TRATAMIENTO} ({f.tratamiento[:50]}) -- actual(outlier)={actual} -> revertido(original)={nuevo_precio}")
    if abs(actual - nuevo_precio) > 0.0001:
        filas_a_actualizar.append({
            "id_proveedor": f.CODPROVEEDOR,
            "id_tratamiento": f.CODIGO_TRATAMIENTO,
            "tratamiento": f.tratamiento,
            "id_municipio": f.id_municipio,
            "pasar_modelo": bool(f.pasar_modelo),
            "coste_medio": nuevo_precio,
        })

print(f"\nPares a revertir: {len(filas_a_actualizar)}")
if not filas_a_actualizar:
    print("Nada que actualizar.")
    sys.exit(0)

confirmar = input("\n¿Revertir estos pares en Supabase? (escribir 'si' para continuar): ")
if confirmar.strip().lower() != "si":
    print("Cancelado -- no se cambió nada.")
    sys.exit(0)

for fila in filas_a_actualizar:
    supabase.table("costo_tratamientos").upsert(fila, on_conflict="id_proveedor,id_tratamiento").execute()
    print(f"  actualizado: proveedor={fila['id_proveedor']} tratamiento={fila['id_tratamiento']} -> coste_medio={fila['coste_medio']}")

print("\nListo.")
