<p align="center">
  <img src="./Images/Steam Games Banner.gif"
 
</p>

<p align="center">
   :small_orange_diamond: <i><u>STACK TECNOLÓGICO:</u></i> :small_orange_diamond:
  
 ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
 ![Steam](https://img.shields.io/badge/steam-%23000000.svg?style=for-the-badge&logo=steam&logoColor=white)
 ![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

  
</p>


<p align="center">
  :small_orange_diamond: <i><u>INTRODUCCIÓN:</u></i> :small_orange_diamond:
</p>

En este proyecto se va a trabajar sobre la plataforam de juegos Steam, desarrollando un rol de Data Engineer para lograr tener un MVP (Minimum Viable Product). En base a 3 datasets iniciales, se va a desarrollar el correspondiente proceso de ETL (Extracción, Transformación y Carga) y de EDA (Análisis Exploratorio de Datos). Además, se va a realizar un  análisis de sentimiento con NLP, y un Modelo de aprendizaje automático, con un sistema de recomendación.

<p align="center">
  :small_orange_diamond: <i><u>ARCHIVOS INICIALES:</u></i> :small_orange_diamond:
</p>

Se tienen 3 datasets:
• australian_user_reviews.json: Conjunto de datos con id de usuarios y sus comentrios de los juegos, su recomendación o no, así como también la url del perfil de usuario y el id del juego.

• australian_users_items.json: Conjunto de datos con información de los juegos, y el tiempo acumulado de juego por cada usuario.

• output_steam_games.json: Conjunto de datos con títulos, géneros, id de los juegos, sus precios y características.

<p align="center">
  :small_orange_diamond: <i><u>PROCESO DE ETL, EDA y FEATURE ENGINEERING:</u></i> :small_orange_diamond:
</p>

Como se mencionó anteriormente, como primera medida se llevó a cabo un proceso de ETL (Extracción, Transformación y Carga), analizando el tipo de dato de cada columna de los distintos datasets, transformándolos cuando fuera necesario, eliminando duplicados, eliminando columnas con valores nulos, desañidando 2 columnas que estaban - justamente - añidadas. También se procedió a eliminar las columnas que no iban a ser de utilidad para el posterior análisis y funcionamiento de api.
Para la realización de la consigna de realizar un un análisis de sentimiento a los comentarios de los usuarios, se introdujo una nueva columna llamada 'sentiment_analysis', la cual sustituye a la columna que originalmente contenía los comentarios de los usuarios. Esta columna clasifica los sentimientos de los comentarios según la siguiente escala:  0 si el sentimiento es negativo, 1 si es neutral o si no hay un comentario asociado,  2 si el sentimiento es positivo. 
Dado que se pedía que se aplicara un análisis de sentimiento con NLP, se utilizó la biblioteca TextBlob, que clasifica la polaridad del texto como positiva, negativa o neutra. 
Se guardaron los datasets limpios en archivos parquet. 
Luego se procedió a la realización del EDA (Análisis Exploratorio de Datos), para identificar los datos necesarios para la posterior realización del modelo de recomendación. Se usaron las librerías Matplotlib y Seaborn para la visualización.
Todo lo antedicho se puede ver en los archivos:  
[ETL_Steam_Games](./Jupyter/ETL_Steam_Games.ipynb)  
[ETL_user_items](./Jupyter/ETL_user_items.ipynb)  
[ETL_users_reviews](./Jupyter/ETL_users_reviews.ipynb)  
[Feature_Engineering_EDA](./Jupyter/Feature_Engineering_EDA.ipynb)  


<p align="center">
  :small_orange_diamond: <i><u>API:</u></i> :small_orange_diamond:
</p>

El desarrollo de la API se realizó usando el framework FastAPI, generando las 5 funciones propuestas para las consultas:
• _PlayTimeGenre:_ Debe devolver año con mas horas jugadas para dicho género.

• _UserForGenre:_ Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

• _UsersRecommend:_ Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.

• _UsersNotRecommend:_ Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.

• _sentiment_analysis( año : int ):_ Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

Posteriormente, se realizó el Modelo de Recomendación Automático, utilizando el sistema de recomendación item-item. Para su realización se utilizó la similitud del coseno, que determina cuán similares son dos conjuntos de datos o elementos, y se calcula utilizando el coseno del ángulo entre los vectores que representan esos datos o elementos.
• _recomendacion_juego( id de producto ):_ Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Lo antedicho se puede observar en el archivo [main](./main.py)  
[FastAPI](http://127.0.0.1:8000/)

<p align="center">
  :small_orange_diamond: <i><u>DEPLOYMENT:</u></i> :small_orange_diamond:
</p>

Luego de verificar que la API funciona a nivel local, se procedió a usar Render para que la misma pueda ser consumida desde la web. Dado que el servicio gratuito de Render consta de poca memoria, se optó por un muestreo porcentual de los Dataframes pertinentes.

#Lo antedicho se puede observar en el siguiente link: [Render](https://pi1-mlops-steam-games.onrender.com)
