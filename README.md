# Trabajo Práctico N°2 - Cicoria - Ricci
## Procesamiento de imágenes 1
Este repositorio contiene el desarrollo del segundo trabajo práctico de la materia **Procesamiento de imágenes 1** de la **Tecnicatura en Inteligencia Artificial**. El objetivo es poder aplicar los conocimientos adquiridos sobre transformaciones geometricas, morfologia, segmentacion por color.

---
### ⚙️ Descripción del entorno de trabajo

El proyecto se implementó utilizando **Python 3.12**. Para garantizar la correcta gestión de dependencias y evitar conflictos, se utilizó un **entorno virtual**.

### 🔧 Configuraciones realizadas

**Paso 1: Crear una carpeta**

Crea una carpeta llamada `TP2-PDI` para alojar todos los archivos del proyecto.

**Paso 2: Descargar los archivos**

Accedé al siguiente repositorio y descargá todos los archivos dentro de la carpeta `TP2-PDI`:

🔗 [Repositorio GitHub](https://github.com/guillericci/TP2_PDI_CICORIA_RICCI)

> ⚠️ **Importante:** Asegurarse de descargar todas las imágenes y archivos `.py`  en la misma carpeta. Esto es fundamental para que el programa funcione correctamente.

**Paso 3: Abrir la terminal**

Desde la terminal, navegá hasta la carpeta del proyecto usando:

```bash
cd ruta/a/TP2-PDI
```

**Paso 4: Crear un entorno virtual**

Creamos un entorno virtual dentro del proyecto:
```bash
python -m venv directorio-env
```
Esto generará una carpeta llamada `directorio-env` que contendrá el entorno virtual.

**Paso 5: Activar el entorno virtual**

En sistemas Windows, se activa con:

```bash
directorio-env\Scripts\activate.bat
```

**Paso 6: Instalar las dependencias**

Primero, actualizamos pip (si es necesario):

```bash
python -m pip install --upgrade pip
```
Luego instalamos las librerías requeridas:

```bash
pip install opencv-python numpy matplotlib os
```
---
### Cómo ejecutar el programa: 

#### Estructura esperada del proyecto:

```bash
TP2-PDI/
├── Resistencias                          # Carpeta con imagenes de las resistencias del problema 2
├── Resistencias_out                      # Carpeta generada por el script del problema 2
├── placa.png                             # Imagen de la placa (problema 1)
├── problema1_PDI.py                     # Script para el Problema 1
├── problema2_PDI.py                     # Script para el Problema 2
├── informe-TP2-PDI                      # Informe del trabajo practico
└── directorio-env/                       # Entorno virtual (no subir a GitHub)
````

#### Ejecución:

Dependiendo del problema que se desee resolver, ejecutar uno de los siguientes comandos:
```bash
python problema1_PDI.py
```
o
```bash
python problema2_PDI.py
```

