# Reemplaza capacidades por completo con lo que hay en medi-match-db/dataset/
# Capaidad_actual.xlsx -- no se conserva ningún dato viejo de la tabla (ni
# capacidad_maxima anterior, ni el consumo histórico heredado de
# CAPACIDAD_CONSUMIDA del dataset original, ver docs/DECISIONES.md,
# 2026-09-13). El motivo: el optimizador (Optimizador_Web_Pesos3.ipynb) solo
# usa CapacidadAnual/capacidad_maxima -- no tiene ningún concepto de
# "consumido" -- así que capacidades pasa a ser un espejo directo de este
# archivo, sin acarrear ninguna otra cosa.
#
# id_municipio también se carga (la migración 024_agregar_id_municipio_
# capacidades.sql tiene que estar aplicada antes de correr esto, o el upsert
# va a fallar porque la columna no existe todavía).
#
# asignados queda en 0 y capacidad_restante = capacidad_maxima para todos --
# pedido explícito del usuario, en conjunto con vaciar la tabla asignaciones
# (025_vaciar_asignaciones.sql): que la app arranque sin ninguna reserva
# previa y con la capacidad llena. Los triggers de asignaciones (013/014)
# siguen funcionando igual de acá en adelante para cualquier reserva nueva.
#
# Las 9.253/9.261 filas del Excel con capacidad_maxima -> se suben tal cual.
# Las 8 filas sin valor (NaN) -> se BORRAN de capacidades (no se inventa un
# número ni se conserva el viejo): ese proveedor+tratamiento queda sin fila
# de capacidad hasta que haya un dato real.
#
# Solo escritura sobre `capacidades` -- no toca proveedores, costo_tratamientos
# ni asignaciones. Usa la service_role key de medi-match-api/.env (no el
# conector MCP de Supabase de esta sesión, que no ve este proyecto).
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

RUTA_EXCEL = "../../medi-match-db/dataset/Capaidad_actual.xlsx"
TAMANO_LOTE = 500

sys.path.insert(0, "../../medi-match-api")
load_dotenv("../../medi-match-api/.env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def descargar_pares_actuales():
    filas = []
    offset = 0
    while True:
        resp = (
            supabase.table("capacidades")
            .select("id_proveedor, id_tratamiento")
            .range(offset, offset + 999)
            .execute()
            .data
        )
        filas.extend(resp)
        if len(resp) < 1000:
            break
        offset += 1000
    return {(f["id_proveedor"], f["id_tratamiento"]) for f in filas}


print("Leyendo Excel...")
df = pd.read_excel(RUTA_EXCEL, sheet_name="Hoja1", skiprows=[1])
df.columns = [c.strip() for c in df.columns]
df["id_proveedor"] = df["id_proveedor"].astype(int)
df["id_tratamiento"] = df["id_tratamiento"].astype(int)

print("Descargando pares existentes...")
pares_actuales = descargar_pares_actuales()
print(f"  capacidades en la base: {len(pares_actuales):,}")

filas_a_subir = []
pares_a_borrar = []
for fila in df.itertuples(index=False):
    clave = (fila.id_proveedor, fila.id_tratamiento)

    if pd.isna(fila.capacidad_maxima):
        if clave in pares_actuales:
            pares_a_borrar.append(clave)
        continue

    capacidad_maxima = int(fila.capacidad_maxima)

    filas_a_subir.append(
        {
            "id_proveedor": clave[0],
            "id_tratamiento": clave[1],
            "id_municipio": fila.id_municipio,
            "capacidad_maxima": capacidad_maxima,
            "asignados": 0,
            "capacidad_restante": capacidad_maxima,
            "estado": "Disponible",
        }
    )

# Pares que estaban en capacidades pero no vinieron en el Excel para nada
# (no debería haber ninguno -- verificado antes -- pero por las dudas se
# borran también, ya que el pedido es que capacidades quede como espejo de
# este archivo, sin nada viejo colgando).
pares_en_excel = {(fila.id_proveedor, fila.id_tratamiento) for fila in df.itertuples(index=False)}
pares_a_borrar += [p for p in pares_actuales if p not in pares_en_excel]
pares_a_borrar = sorted(set(pares_a_borrar))

print(f"\nFilas a actualizar: {len(filas_a_subir):,}")
print(f"Filas a borrar (sin capacidad_maxima en el Excel, o no vinieron en el Excel): {len(pares_a_borrar):,}")

confirmar = input("\n¿Subir estos cambios a Supabase? (escribir 'si' para continuar): ")
if confirmar.strip().lower() != "si":
    print("Cancelado -- no se subió nada.")
    sys.exit(0)

if pares_a_borrar:
    print("\nBorrando pares sin dato nuevo...")
    for id_proveedor, id_tratamiento in pares_a_borrar:
        supabase.table("capacidades").delete().eq("id_proveedor", id_proveedor).eq(
            "id_tratamiento", id_tratamiento
        ).execute()
    print(f"  {len(pares_a_borrar):,} borradas.")

print("\nSubiendo en lotes...")
for i in range(0, len(filas_a_subir), TAMANO_LOTE):
    lote = filas_a_subir[i : i + TAMANO_LOTE]
    supabase.table("capacidades").upsert(lote, on_conflict="id_proveedor,id_tratamiento").execute()
    print(f"  {min(i + TAMANO_LOTE, len(filas_a_subir)):,}/{len(filas_a_subir):,}")

print("\nListo.")
