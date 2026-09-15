# Exporta las 6 tablas reales de Supabase (medi-match-db) a un solo Excel,
# usando las credenciales de medi-match-api/.env -- NO el conector MCP de esta
# sesión, que no ve el proyecto real (ver docs/DECISIONES.md, 2026-08-25).
# Solo lectura -- ningún insert/update/delete contra producción.
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, "../../medi-match-api")
load_dotenv("../../medi-match-api/.env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

TABLAS = ["proveedores", "costo_tratamientos", "capacidades", "clientes", "gestores", "asignaciones"]


def descargar_tabla(nombre):
    filas = []
    offset = 0
    tam_pagina = 1000
    while True:
        resp = supabase.table(nombre).select("*").range(offset, offset + tam_pagina - 1).execute()
        lote = resp.data
        filas.extend(lote)
        if len(lote) < tam_pagina:
            break
        offset += tam_pagina
    return pd.DataFrame(filas)


tablas = {}
for nombre in TABLAS:
    df = descargar_tabla(nombre)
    tablas[nombre] = df
    print(f"{nombre}: {len(df):,} filas, columnas: {list(df.columns)}")

os.makedirs("_export_supabase_raw", exist_ok=True)
for nombre, df in tablas.items():
    df.to_pickle(f"_export_supabase_raw/{nombre}.pkl")

print("\nDescarga completa, guardada en _export_supabase_raw/*.pkl (insumo para el excel maestro)")
