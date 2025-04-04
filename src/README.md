# Ejecución de Tests Funcionales del Modelo para Inmobiliaria

### Paso 0: Ingrese al Escritorio remoto

### Paso 1: Fork del Repositorio Original

En el navegador, inicie sesión en Github. Luego, vaya al enlace del proyecto original (https://github.com/darkhasl/model-inmobiliaria) y dé click al botón "Fork". Esto copiará todo el proyecto en su usuario de Github.


### Paso 2: Levantar el contenedor de Python

```
docker run -it --rm -p 8888:8888 jupyter/pyspark-notebook
```


### Paso 3: Configurar git

Abra una Terminal en JupyterLab e ingrese los siguientes comandos

```
git config --global user.name "<USER>"
git config --global user.email <CORREO>
```


### Paso 4: Clonar el Proyecto desde su propio Github

```
git clone https://github.com/<USER>/model-inmobiliaria.git
```


### Paso 5: Instalar los pre-requisitos

```
cd model-inmobiliaria/

pip install -r requirements.txt
```


### Paso 6: Ejecutar las pruebas en el entorno

```
cd src

python make_dataset.py

python train.py

python evaluate.py

python predict.py

cd ..
```

![image](https://github.com/user-attachments/assets/dd28f2a6-fbf2-4a45-acc1-ed3b97fd3a44)

### Paso 7: Guardar los cambios en el Repo

```
git add .

git commit -m "Pruebas Finalizadas"

git push

```

Ingrese su usuario y Personal Access Token de Github. Puede revisar que los cambios se hayan guardado en el repositorio. Luego, puede finalizar JupyterLab ("File" => "Shut Down").


