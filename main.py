import os
from datetime import date
from typing import List, Literal, Optional

import bcrypt
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
    id_gestor: str
    fecha_servicio: date = Field(default_factory=date.today)
    tipo_servicio: Literal["programado", "urgencia"] = "programado"
    observaciones: Optional[str] = None
    id_reserva: Optional[str] = None  # si viene, es una edición, no una reserva nueva


class ResponseReserva(BaseModel):
    status: str
    mensaje: str
    id_reserva: Optional[str] = None


class Cliente(BaseModel):
    id_cliente: str
    poliza: str
    documento: str
    nombre_completo: str
    tipo_usuario: str


class LoginRequest(BaseModel):
    email: str
    password: str


class Gestor(BaseModel):
    id_gestor: str
    nombre: str
    email: str


class AsignacionDetalle(BaseModel):
    id_reserva: str
    id_proveedor: int
    fecha_reserva: str
    fecha_servicio: str
    tipo_servicio: str
    tratamiento: str
    observaciones: Optional[str] = None
    nombre_proveedor: str
    ciudad: str
    id_cliente: str
    nombre_cliente: str
    poliza: str
    documento: str
    tipo_usuario: str
    id_gestor: str
    nombre_gestor: str


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


@app.post("/login", response_model=Gestor)
def login(request: LoginRequest) -> Gestor:
    filas = (
        supabase.table("gestores")
        .select("id_gestor, nombre, email, password_hash")
        .eq("email", request.email)
        .execute()
        .data
    )
    if not filas or not bcrypt.checkpw(
        request.password.encode("utf-8"), filas[0]["password_hash"].encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    gestor = filas[0]
    return Gestor(id_gestor=gestor["id_gestor"], nombre=gestor["nombre"], email=gestor["email"])


@app.get("/clientes", response_model=List[Cliente])
def buscar_clientes(q: str = Query(..., min_length=2)) -> List[Cliente]:
    # Autocompleta el formulario: un solo cuadro de búsqueda que matchea por
    # póliza, documento o nombre (lo que sea que el usuario haya tipeado).
    filas = (
        supabase.table("clientes")
        .select("id_cliente, poliza, documento, nombre_completo, tipo_usuario")
        .or_(f"poliza.ilike.%{q}%,documento.ilike.%{q}%,nombre_completo.ilike.%{q}%")
        .limit(10)
        .execute()
        .data
    )
    return filas


@app.post("/recomendar", response_model=List[ResponseProveedor])
def recomendar(request: RequestRecomendacion) -> List[ResponseProveedor]:
    proveedores = (
        supabase.table("proveedores")
        .select("id_proveedor, nombre_proveedor, valoracion")
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
        .select("id_proveedor, id_tratamiento, coste_medio")
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

    # La capacidad es por par (proveedor, tratamiento), no un pool compartido
    # por todo lo que ofrece el proveedor.
    ids_tratamiento = list({fila["id_tratamiento"] for fila in catalogo_por_proveedor.values()})
    capacidades_filas = (
        supabase.table("capacidades")
        .select("id_proveedor, id_tratamiento, capacidad_restante")
        .in_("id_proveedor", list(catalogo_por_proveedor.keys()))
        .in_("id_tratamiento", ids_tratamiento)
        .execute()
        .data
    )
    capacidad_por_par = {
        (c["id_proveedor"], c["id_tratamiento"]): c["capacidad_restante"] for c in capacidades_filas
    }

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
            capacidad_restante=capacidad_por_par.get((p["id_proveedor"], fila["id_tratamiento"]), 0),
            indice_ranking=indice,
        )
        for (p, fila), indice in zip(combinados, indices)
    ]
    return sorted(respuesta, key=lambda p: p.indice_ranking, reverse=True)


@app.post("/reservar", response_model=ResponseReserva)
def reservar(request: RequestReserva) -> ResponseReserva:
    # asignaciones necesita el código del tratamiento (capacidades está keyed
    # por id_tratamiento, no por el texto) — se resuelve a partir de
    # (id_proveedor, tratamiento) contra el catálogo, sin pedirle ese código
    # al frontend.
    fila_tratamiento = (
        supabase.table("costo_tratamientos")
        .select("id_tratamiento, id_ciudad")
        .eq("id_proveedor", request.id_proveedor)
        .eq("tratamiento", request.tratamiento)
        .limit(1)
        .execute()
        .data
    )
    if not fila_tratamiento:
        return ResponseReserva(status="error", mensaje="Ese proveedor no ofrece ese tratamiento")
    id_tratamiento = fila_tratamiento[0]["id_tratamiento"]
    id_ciudad = fila_tratamiento[0]["id_ciudad"]

    es_edicion = request.id_reserva is not None
    par_anterior = None

    if es_edicion:
        actual = (
            supabase.table("asignaciones")
            .select("id_proveedor, id_tratamiento")
            .eq("id_reserva", request.id_reserva)
            .eq("estado", "activo")
            .execute()
            .data
        )
        if not actual:
            return ResponseReserva(status="error", mensaje="La asignación no existe")
        par_anterior = (actual[0]["id_proveedor"], actual[0]["id_tratamiento"])

    # Si es una edición y el par (proveedor, tratamiento) no cambió, no hace
    # falta chequear cupo (no se está consumiendo capacidad nueva). El
    # trigger de UPDATE en medi-match-db se encarga de mover la capacidad
    # entre pares si sí cambió; el de INSERT, de descontarla en una reserva
    # nueva.
    cambia_par = not es_edicion or par_anterior != (request.id_proveedor, id_tratamiento)
    if cambia_par:
        capacidad = (
            supabase.table("capacidades")
            .select("capacidad_restante")
            .eq("id_proveedor", request.id_proveedor)
            .eq("id_tratamiento", id_tratamiento)
            .execute()
        )
        if not capacidad.data or capacidad.data[0]["capacidad_restante"] <= 0:
            return ResponseReserva(status="error", mensaje="Sin cupos disponibles")

    payload = {
        "id_cliente": request.id_cliente,
        "id_proveedor": request.id_proveedor,
        "id_tratamiento": id_tratamiento,
        "id_ciudad": id_ciudad,
        "tratamiento": request.tratamiento,
        "id_gestor": request.id_gestor,
        "fecha_servicio": request.fecha_servicio.isoformat(),
        "tipo_servicio": request.tipo_servicio,
        "observaciones": request.observaciones,
    }

    if es_edicion:
        supabase.table("asignaciones").update(payload).eq("id_reserva", request.id_reserva).execute()
        return ResponseReserva(
            status="success", mensaje="Asignación actualizada", id_reserva=request.id_reserva
        )

    fila = supabase.table("asignaciones").insert(payload).execute().data[0]
    return ResponseReserva(status="success", mensaje="Asignado", id_reserva=fila["id_reserva"])


@app.get("/asignaciones", response_model=List[AsignacionDetalle])
def listar_asignaciones(id_gestor: Optional[str] = None) -> List[AsignacionDetalle]:
    query = (
        supabase.table("asignaciones")
        .select(
            "id_reserva, id_proveedor, fecha_reserva, fecha_servicio, tipo_servicio,"
            " tratamiento, observaciones, id_cliente, id_gestor,"
            " proveedores(nombre_proveedor, ciudad),"
            " clientes(nombre_completo, poliza, documento, tipo_usuario),"
            " gestores(nombre)"
        )
        .eq("estado", "activo")
        .order("fecha_reserva", desc=True)
    )
    if id_gestor:
        query = query.eq("id_gestor", id_gestor)

    filas = query.execute().data
    return [
        AsignacionDetalle(
            id_reserva=f["id_reserva"],
            id_proveedor=f["id_proveedor"],
            fecha_reserva=f["fecha_reserva"],
            fecha_servicio=f["fecha_servicio"],
            tipo_servicio=f["tipo_servicio"],
            tratamiento=f["tratamiento"],
            observaciones=f["observaciones"],
            nombre_proveedor=f["proveedores"]["nombre_proveedor"],
            ciudad=f["proveedores"]["ciudad"],
            id_cliente=f["id_cliente"],
            nombre_cliente=f["clientes"]["nombre_completo"],
            poliza=f["clientes"]["poliza"],
            documento=f["clientes"]["documento"],
            tipo_usuario=f["clientes"]["tipo_usuario"],
            id_gestor=f["id_gestor"],
            nombre_gestor=f["gestores"]["nombre"],
        )
        for f in filas
    ]


@app.delete("/asignaciones/{id_reserva}", response_model=ResponseReserva)
def eliminar_asignacion(id_reserva: str) -> ResponseReserva:
    # Soft delete: no se borra la fila (queda como auditoría), solo se marca
    # estado='eliminado'. El trigger de UPDATE en medi-match-db libera la
    # capacidad automáticamente en esa transición.
    supabase.table("asignaciones").update({"estado": "eliminado"}).eq("id_reserva", id_reserva).execute()
    return ResponseReserva(status="success", mensaje="Asignación eliminada")
