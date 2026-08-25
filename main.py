import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy import stats
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(title="MediMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestRecomendacion(BaseModel):
    ciudad: str
    tratamiento: str
    umbral_valoracion: float = 0


class ResponseProveedor(BaseModel):
    id_proveedor: int
    nombre_proveedor: str
    coste_estimado: float
    valoracion: float
    capacidad_restante: int
    indice_ranking: float


class RequestReserva(BaseModel):
    id_proveedor: int
    ciudad: str
    tratamiento: str
    id_cliente: str


class ResponseReserva(BaseModel):
    status: str
    mensaje: str


def calcular_indice_ranking(costos: List[float], valoraciones: List[float]) -> List[float]:
    costos_arr = np.array(costos, dtype=float)
    valoraciones_arr = np.array(valoraciones, dtype=float)

    costo_z = stats.zscore(costos_arr) if costos_arr.std() > 0 else np.zeros_like(costos_arr)
    valoracion_z = (
        stats.zscore(valoraciones_arr) if valoraciones_arr.std() > 0 else np.zeros_like(valoraciones_arr)
    )

    # Afinidad: a menor costo y mayor valoración, mejor índice. Se combinan con
    # igual peso y se pasan por una sigmoide para acotar el resultado a (0, 1).
    score = 0.5 * (-costo_z) + 0.5 * valoracion_z
    return (1 / (1 + np.exp(-score))).tolist()


@app.get("/ciudades", response_model=List[str])
def listar_ciudades() -> List[str]:
    filas = supabase.table("proveedores").select("ciudad").execute().data
    return sorted({f["ciudad"] for f in filas})


@app.get("/tratamientos", response_model=List[str])
def listar_tratamientos(ciudad: str = Query(...)) -> List[str]:
    ids_proveedor = [
        p["id_proveedor"]
        for p in supabase.table("proveedores").select("id_proveedor").eq("ciudad", ciudad).execute().data
    ]
    if not ids_proveedor:
        return []

    filas = (
        supabase.table("costo_tratamientos")
        .select("tratamiento")
        .in_("id_proveedor", ids_proveedor)
        .execute()
        .data
    )
    return sorted({f["tratamiento"] for f in filas})


@app.post("/recomendar", response_model=List[ResponseProveedor])
def recomendar(request: RequestRecomendacion) -> List[ResponseProveedor]:
    proveedores = (
        supabase.table("proveedores")
        .select("id_proveedor, nombre_proveedor, valoracion, capacidades(capacidad_restante)")
        .eq("ciudad", request.ciudad)
        .gte("valoracion", request.umbral_valoracion)
        .execute()
        .data
    )
    if not proveedores:
        return []

    ids_proveedor = [p["id_proveedor"] for p in proveedores]
    catalogo = (
        supabase.table("costo_tratamientos")
        .select("id_proveedor, coste_medio")
        .in_("id_proveedor", ids_proveedor)
        .eq("tratamiento", request.tratamiento)
        .execute()
        .data
    )
    if not catalogo:
        return []

    proveedores_por_id = {p["id_proveedor"]: p for p in proveedores}
    # Si un proveedor ofrece el tratamiento buscado a más de un costo (varias
    # líneas del dataset agrupadas distinto), nos quedamos con la más barata.
    catalogo_por_proveedor = {}
    for fila in catalogo:
        actual = catalogo_por_proveedor.get(fila["id_proveedor"])
        if actual is None or fila["coste_medio"] < actual["coste_medio"]:
            catalogo_por_proveedor[fila["id_proveedor"]] = fila

    combinados = [
        (proveedores_por_id[id_proveedor], fila)
        for id_proveedor, fila in catalogo_por_proveedor.items()
    ]

    indices = calcular_indice_ranking(
        [fila["coste_medio"] for _, fila in combinados],
        [p["valoracion"] for p, _ in combinados],
    )

    respuesta = [
        ResponseProveedor(
            id_proveedor=p["id_proveedor"],
            nombre_proveedor=p["nombre_proveedor"],
            coste_estimado=fila["coste_medio"],
            valoracion=p["valoracion"],
            capacidad_restante=(p["capacidades"] or {}).get("capacidad_restante", 0),
            indice_ranking=indice,
        )
        for (p, fila), indice in zip(combinados, indices)
    ]
    return sorted(respuesta, key=lambda p: p.indice_ranking, reverse=True)


@app.post("/reservar", response_model=ResponseReserva)
def reservar(request: RequestReserva) -> ResponseReserva:
    capacidad = (
        supabase.table("capacidades")
        .select("capacidad_restante")
        .eq("id_proveedor", request.id_proveedor)
        .execute()
    )
    if not capacidad.data or capacidad.data[0]["capacidad_restante"] <= 0:
        return ResponseReserva(status="error", mensaje="Sin cupos disponibles")

    # El trigger de medi-match-db descuenta capacidad automáticamente al insertar aquí.
    supabase.table("asignaciones").insert(
        {"id_cliente": request.id_cliente, "id_proveedor": request.id_proveedor}
    ).execute()

    return ResponseReserva(status="success", mensaje="Asignado")
