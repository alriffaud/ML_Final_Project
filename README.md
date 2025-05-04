# Machine Learning - Final Project

## Data Preprocessing

### Obtención de los archivos de datos

Nota: Esta parte es opcional, ya que los archivos de salida ya están disponibles en la carpeta Data_Preprocessing/data.

Primero, se comienza por extraer las tablas **evaluation_quizzes**, **evaluation_quiz_questions** y **evaluation_quiz_corrections** del archivo *quizzes.sql*. Para ello se ejecuta el siguiente comando en la terminal, luego de haber instalado y configurado MySQL, y estando ubicado en la carpeta principal del proyecto:

```bash
$ echo "CREATE DATABASE quizzes;" | mysql -uroot -p
Enter password:
$ cat Data_Preprocessing/data/quizzes.sql | mysql -uroot -p
Enter password:
```	
Esto creará la base de datos quizzes y cargará el archivo quizzes.sql en ella. Luego, para ejecutar el script de Python, se debe tener instalado Python 3.8 o superior y las librerías requeridas. Para instalar las librerías requeridas, ejecutar el siguiente comando en la terminal:

```bash
$ pip install -r requirements.txt
```
Una vez instaladas las librerías, se puede ejecutar el script de Python (archivo export_to_CSV.py) con el siguiente comando desde la carpeta Data_Preprocessing:

```bash
$ python3 export_to_CSV.py
```

Esto generará los archivos de salida en la carpeta Data_Preprocessing/data.