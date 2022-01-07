# Author Monroy Velázquez Alejandra Sarahí
# -----------------------------------------------------------------------------
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from types import MethodDescriptorType
from pandas.io.formats.format import TextAdjustment
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from apyori import apriori
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
from io import BytesIO
import os
from flask import Flask, render_template, url_for, request, Response, flash, redirect, session
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use('Agg')

UPLOAD_FOLDER = 'uploads'

# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "3d6f45a5fc12445dbac2f59c3b6c7cb1"
# -----------------------------------------------------------------------------


@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------------------------------------------------------

# Helpers


def upload_files():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    route = "uploads/{}".format(filename)
    df = pd.read_csv(route)
    session['my_var'] = route
    return df, filename


def compute(plt, fignum):
    # run plt.plot, plt.title, etc.
    plt.figure(fignum)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    # figfile.getvalue()  extracts string (stream of bytes)
    figdata_png = base64.b64encode(figfile.getvalue())
    return Response(figfile.getvalue(), mimetype='image/png')

# -----------------------------------------------------------------------------


@app.route('/AnalisisExploratorio', methods=['GET', 'POST'])
def eda():
    if request.method == 'POST':
        renderizar = True

        df, filename = upload_files()

        firstHead = df.head()
        shape = df.shape
        types = pd.DataFrame(df.dtypes)
        missing = pd.DataFrame(df.isnull().sum())
        resume = df.describe()
        describe = df.describe(include='object')
        correlation = df.corr()

        return render_template('analisis.html', renderizar=renderizar, filename=filename, tables=[firstHead.to_html(classes='data')], titles=firstHead.columns.values, shape=shape, types=types, tablesTypes=[types.to_html(classes='data')], titlesTypes=types.columns.values, tablesMissing=[missing.to_html(classes='data')], titlesMissing=missing.columns.values, tablesResume=[resume.to_html(classes='data')], titlesResume=resume.columns.values, tablesDescribe=[describe.to_html(classes='data')], titlesDescribe=describe.columns.values, tablesCorr=[correlation.to_html(classes='data')], titlesCorr=correlation.columns.values)
    else:
        renderizar = False
        return render_template('analisis.html', renderizar=renderizar)


@app.route('/plotDistribucion.png')
def dibuja_grafico():
    filename = session.get('my_var', None)
    df = pd.read_csv(filename)
    histogram = df.hist(figsize=(14, 14), xrot=45)
    output = io.BytesIO()
    plt.legend()
    plt.savefig(output, format='png')
    plt.close()
    return Response(output.getvalue(), mimetype='image/png')

# -----------------------------------------------------------------------------


@app.route('/SeleccionCaracteristicas')
def selectCaract():
    return render_template('caracteristicas.html')


@app.route('/ACD', methods=['GET', 'POST'])
def acd():
    if request.method == 'POST' and 'variables' not in request.form:
        renderizar = True

        df, filename = upload_files()

        firstHead = df.head()
        shape = df.shape
        correlation = df.corr()

        return render_template('acd.html', renderizar=renderizar, filename=filename, shape=shape, tables=[firstHead.to_html(classes='data')], titles=firstHead.columns.values, tablesCorr=[correlation.to_html(classes='data')], titlesCorr=correlation.columns.values)

    elif request.method == 'POST' and 'variables' in request.form:
        route = session['my_var']
        df = pd.read_csv(route)
        variables = request.form['variables']
        eliminar = variables.split(',')
        newSet = df.drop(columns=eliminar)
        newSet = newSet.head(10)

        return render_template('nuevoSet.html', tables=[newSet.to_html(classes='data')], titles=newSet.columns.values)
    else:
        renderizar = False
        return render_template('acd.html')


@app.route('/plotCorrelacion.png')
def heat_map():
    filename = session.get('my_var', None)
    df = pd.read_csv(filename)
    plt.figure(2, figsize=(14, 14))
    MatrizInf = np.triu(df.corr())
    sns.heatmap(df.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
    return compute(plt, 2)


@app.route('/PCA', methods=['GET', 'POST'])
def pca():
    if request.method == 'POST' and 'variables' not in request.form:
        renderizar = True

        df, filename = upload_files()

        firstHead = df.head()
        shape = df.shape

        normalizar = StandardScaler()
        normalizar.fit(df)
        MNormalizada = normalizar.transform(df)
        normalized = pd.DataFrame(MNormalizada, columns=df.columns)
        pca = PCA(n_components=10)
        pca.fit(MNormalizada)
        variance = pca.explained_variance_ratio_
        tableVariance = pd.DataFrame(variance)

        components = pd.DataFrame(abs(pca.components_), columns=df.columns)

        return render_template('pca.html', renderizar=renderizar, filename=filename, shape=shape, tables=[firstHead.to_html(classes='data')], titles=firstHead.columns.values, tableNormalized=[normalized.to_html(classes='data')], titlesNormalized=normalized.columns.values, tableComponents=[components.to_html(classes='data')], titlesComponents=components.columns.values, tableVariance=[tableVariance.to_html(classes='data')])

    elif request.method == 'POST' and 'variables' in request.form:
        route = session['my_var']
        df = pd.read_csv(route)
        variables = request.form['variables']
        eliminar = variables.split(',')
        newSet = df.drop(columns=eliminar)
        newSet = newSet.head(10)

        return render_template('nuevoSet.html', tables=[newSet.to_html(classes='data')], titles=newSet.columns.values)
    else:
        renderizar = False
        return render_template('pca.html')


@app.route('/plotVarianza.png')
def variance_graph():
    filename = session.get('my_var', None)
    df = pd.read_csv(filename)
    normalizar = StandardScaler()
    normalizar.fit(df)
    MNormalizada = normalizar.transform(df)
    normalized = pd.DataFrame(MNormalizada, columns=df.columns)
    pca = PCA(n_components=10)
    pca.fit(MNormalizada)
    # print(pca.components_)
    variance = pca.explained_variance_ratio_
    plt.plot(np.cumsum(variance))
    output = io.BytesIO()
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza acumulada')
    plt.grid()
    plt.savefig(output, format='png')
    plt.close()
    return Response(output.getvalue(), mimetype='image/png')

# -----------------------------------------------------------------------------


@app.route('/Clusterizacion')
def cluster():
    return render_template('clusters.html')


@app.route('/Jerarquico')
def clusterj():
    return render_template('clusters.html')


@app.route('/Particional')
def clusterp():
    return render_template('clusters.html')


# -----------------------------------------------------------------------------


@ app.route('/ReglasAsociacion', methods=['GET', 'POST'])
def reglas():
    if request.method == 'POST':
        renderizar = True

        df, filename = upload_files()

        support = float(request.form['support'])
        confidence = float(request.form['confidence'])
        lift = float(request.form['lift'])
        supportp = (support*100)
        confidencep = (confidence*100)

        firstHead = df.head()
        shape = df.shape
        transactions = df.values.reshape(-1).tolist()
        Lista = listOfFrequency(transactions)
        # Lista = pd.DataFrame(Lista)
        listTransactions = df.stack().groupby(level=0).apply(list).tolist()
        rules = apriori(listTransactions, min_support=support,
                        min_confidence=confidence, min_lift=lift)
        results = list(rules)
        numRules = len(results)
        rulesDf = pd.DataFrame(results)

        return render_template('reglas.html', renderizar=renderizar, filename=filename, support=supportp, confidence=confidencep, lift=lift, tables=[firstHead.to_html(classes='data')], titles=firstHead.columns.values, tablesLista=[Lista.to_html(classes='data')], titlesLista=Lista.columns.values, numRules=numRules, tablesRules=[rulesDf.to_html(classes='data')], rulesDf=rulesDf)
    else:
        renderizar = False
        return render_template('reglas.html')


def listOfFrequency(transactions):
    Lista = pd.DataFrame(transactions)
    Lista['Frecuencia'] = 0
    Lista = Lista.groupby(by=[0], as_index=False).count(
    ).sort_values(by=['Frecuencia'], ascending=True)  # Frequency
    Lista['Porcentaje'] = (Lista['Frecuencia'] /
                           Lista['Frecuencia'].sum())  # Porcentaje
    Lista = Lista.rename(columns={0: 'Item'})
    return Lista


@app.route('/plotBarras.png')
def barras():
    filename = session.get('my_var', None)
    df = pd.read_csv(filename)
    transactions = df.values.reshape(-1).tolist()
    Lista = listOfFrequency(transactions)
    plt.figure(figsize=(16, 20))
    plt.ylabel('Item')
    plt.xlabel('Frecuencia')
    plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='purple')
    output = io.BytesIO()
    plt.legend()
    plt.savefig(output, format='png')
    plt.close()
    return Response(output.getvalue(), mimetype='image/png')

# -----------------------------------------------------------------------------


@ app.route('/Arboles')
def arboles():
    return render_template('arboles.html')


@ app.route('/Pronostico', methods=['GET', 'POST'])
def pronostico():
    if request.method == 'POST':
        renderizar = True

        df, filename = upload_files()
        predict = request.form['predict']
        pronostic = request.form['pronostic']
        test = float(request.form['test'])
        depth = int(request.form['depth'])
        samplesSplit = int(request.form['samples-split'])
        samplesLeaf = int(request.form['samples-leaf'])
        newPr = request.form['valores']

        predict = predict.split(',')
        var2 = tuple(map(float, newPr.split(',')))
        var2 = list(var2)

        key_list = predict
        value_list = var2

        dict_from_list = dict(zip(key_list, value_list))

        df2 = pd.DataFrame([dict_from_list])

        X = np.array(df[predict])
        Y = np.array(df[pronostic])
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
            X, Y, test_size=test, random_state=1234, shuffle=True)

        PronosticoAD = DecisionTreeRegressor()
        PronosticoAD.fit(X_train, Y_train)
        PronosticoAD = DecisionTreeRegressor(
            max_depth=depth, min_samples_split=samplesSplit, min_samples_leaf=samplesLeaf)
        PronosticoAD.fit(X_train, Y_train)

        # se genera el pronostico
        Y_Pronostico = PronosticoAD.predict(X_test)

        criterio = PronosticoAD.criterion
        importancia = PronosticoAD.feature_importances_
        mae = mean_absolute_error(Y_test, Y_Pronostico)
        mse = mean_squared_error(Y_test, Y_Pronostico)
        rmse = mean_squared_error(Y_test, Y_Pronostico, squared=False)
        score = r2_score(Y_test, Y_Pronostico)

        importanciaTable = pd.DataFrame({'Variable': list(
            df[predict]), 'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)

        newPronostico = PronosticoAD.predict(df2)
        newPronostico = float(newPronostico)

        return render_template('pronostico.html', renderizar=renderizar, filename=filename, predict=predict, pronostic=pronostic, test=test, depth=depth, samplesSplit=samplesSplit, samplesLeaf=samplesLeaf, criterio=criterio, importancia=importancia, mae=mae, mse=mse, rmse=rmse, score=score, tableImportancia=[importanciaTable.to_html(classes='data')], newPronostico=newPronostico)

    else:
        renderizar = False
        return render_template('pronostico.html', renderizar=renderizar)


@ app.route('/Clasificacion', methods=['GET', 'POST'])
def clasificacion():
    if request.method == 'POST':
        renderizar = True

        df, filename = upload_files()
        predict = request.form['predict']
        print(predict)
        pronostic = request.form['pronostic']
        test = float(request.form['test'])
        depth = int(request.form['depth'])
        samplesSplit = int(request.form['samples-split'])
        samplesLeaf = int(request.form['samples-leaf'])
        newPr = request.form['valores']

        predict = predict.split(',')
        var2 = tuple(map(float, newPr.split(',')))
        var2 = list(var2)

        key_list = predict
        value_list = var2
        dict_from_list = dict(zip(key_list, value_list))
        df2 = pd.DataFrame([dict_from_list])

        X = np.array(df[predict])
        Y = np.array(df[pronostic])
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
            X, Y, test_size=test, random_state=0, shuffle=True)

        ClasificacionAD = DecisionTreeClassifier(
            max_depth=depth, min_samples_split=samplesSplit, min_samples_leaf=samplesLeaf)
        ClasificacionAD.fit(X_train, Y_train)

        Y_Clasificacion = ClasificacionAD.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(
        ), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación'])

        criterio = ClasificacionAD.criterion
        exactitud = ClasificacionAD.score(X_validation, Y_validation)
        score = ClasificacionAD.score(X_validation, Y_validation)

        importanciaTable = pd.DataFrame({'Variable': list(
            df[predict]), 'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)

        newPronostico = ClasificacionAD.predict(df2)
        newPronostico = str(newPronostico)

        return render_template('clasificacion.html', renderizar=renderizar, filename=filename, predict=predict, pronostic=pronostic, test=test, depth=depth, samplesSplit=samplesSplit, samplesLeaf=samplesLeaf, criterio=criterio, score=score, exactitud=exactitud, tableImportancia=[importanciaTable.to_html(classes='data')], newPronostico=newPronostico, tableMatriz=[Matriz_Clasificacion.to_html(classes='data')])

    else:
        renderizar = False
        return render_template('clasificacion.html', renderizar=renderizar)

# -----------------------------------------------------------------------------


@ app.route('/Contacto')
def contacto():
    return render_template('contacto.html')


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(threading=False, debug=True)
