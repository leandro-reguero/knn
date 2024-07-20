https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv
```

#### Paso 1: Carga del conjunto de datos

Debemos cargar los dos ficheros y almacenarlos en dos estructuras de datos (DataFrames de Pandas) separadas. Por un lado, tendremos almacenada la información de las películas y sus créditos.

#### Paso 2: Creación de una base de datos

Crea una base de datos para almacenar los dos DataFrames en tablas distintas. A continuación, une las dos tablas con SQL (e intégralo con Python) para generar una tercera tabla que contenga información de ambas unificada. La clave a través de la cual se puede hacer la unión es el título de la película (`titulo`).

Ahora, limpia la tabla generada y deja solo las siguientes columnas:

- `movie_id`
- `title`
- `overview`
- `genres`
- `keywords`
- `cast`
- `crew`

#### Paso 3: Transforma los datos

Como puedes ver, hay algunas columnas con formato JSON. De cada uno de los JSONs, selecciona el atributo `name` y reemplaza las columnas `genres` y `keywords`. Para la columna `cast`, selecciona los tres primeros nombres.

Las únicas columnas que quedan por modificar son `crew` (equipo) y `overview` (resumen). Para la primera columna, transfórmala para que contenga el nombre del director. Para la segunda, conviértela en una lista.

Una vez hayamos terminado de procesar las columnas y que el modelo de recomendación no se confunda, por ejemplo, entre *Jennifer Aniston* y *Jennifer Conelly*, quitaremos los espacios entre las palabras. Aplica esta función a las columnas `genres`, `cast`, `crew` y `keywords`.

Por último, reduciremos nuestro conjunto de datos combinando todas nuestras columnas convertidas anteriores en una sola columna llamada `tags` (que crearemos). Esta columna ahora tendrá todos los elementos separados por comas y luego las reemplazaremos por espacios en blanco. Debería quedar algo así:

```py
new_df["tags"][0]

>>>>"In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron"
```

#### Paso 4: Construye un KNN

Para resolver este problema crearemos nosotros nuestro propio KNN. Lo primero de todo es vectorizar el texto siguiendo los mismos pasos que aprendiste en la lección anterior.

Una vez lo hayas hecho, tendríamos que elegir una distancia para comparar texto. En este módulo hemos visto algunas, y la única compatible con lo que queremos hacer es la `distancia coseno`:

```py
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
```

Con este código podremos ver la similitud existente entre nuestros vectores (representaciones vectoriales de la columna `tags`).

Finalmente, podemos diseñar nuestra función de similitud basada en la distancia del coseno. Nuestra propuesta es la siguiente:

```py
def recommend(movie):
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse = True , key = lambda x: x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
```

De tal forma que devolveríamos las 5 películas más similares a la que introduzcamos en el título. Podríamos utilizarla como sigue:

```py
recommend("Introduce una película")
```

> Nota: También incorporamos muestras de solución en `./solution.ipynb` que te sugerimos honestamente que solo uses si estás atascado por más de 30 minutos o si ya has terminado y quieres compararlo con tu enfoque.