from flask import Flask, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
import mysql.connector as sql
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import sklearn.cross_validation
import statsmodels.api as sm
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.tools.plotting import scatter_matrix
import StringIO
import base64
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

app = Flask(__name__)

db_connection = sql.connect(host='mailman82.mysql.pythonanywhere-services.com', database='mailman82$CHS_HOUSING', user='mailman82', password='Celtics17')
db_cursor = db_connection.cursor()


df = pd.read_sql('SELECT * FROM data', con=db_connection)
df2 = pd.read_sql('SELECT * FROM categories', con=db_connection)
df_TWN = pd.get_dummies(df['TWN'])
df_HDS = pd.get_dummies(df['HDS'])
df_final = pd.concat([df[['YR','MSP','PWP','HIR','SCH','INF','WSC','NCR']],df_TWN,df_HDS],axis=1)

def bestR2(X, y, X_train, X_test, y_train, y_test):
    alphas = np.logspace(-5, -1, 100)
    train_errors=[]
    coeffs=[]
    scores=[]
    for alpha in alphas:
        regr = Lasso(alpha=alpha)
        # Train the model using the training sets
        regr.fit(X_train, y_train)
        train_errors.append(regr.score(X_train,y_train))
        scores.append(regr.score(X_test,y_test))
        coeffs.append(regr.coef_)
    alpha_optim=alphas[np.argmax(scores)]
    regr = Lasso(alpha=alpha_optim)
    scores = cross_val_score(regr, X, y, cv=5)

    img = StringIO.StringIO()
    plt.figure(1)
    plt.ylim([-1,1])
    plt.xlabel('lambda')
    plt.ylabel('R^2')
    plt.title('Performance on 5 folds with lambda=' + str(alpha_optim))
    plt.bar(range(1,6),scores)
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue())
    return scores, alpha_optim, plot_url

def GSCV(X,y):
    alphas = np.linspace(0.0000001,0.0001,500)
    model = Lasso()
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas),cv=3)
    grid.fit(X,y)
    gbs = grid.best_score_
    gbe = grid.best_estimator_.alpha
    return grid, gbs, gbe

def run_cv(data,X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()
    coeffs=[]
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
        coeffs.append(clf.coef_)
    coeffs_avgd = [(coeffs[0][i] + coeffs[1][i] + coeffs[2][i] + coeffs[3][i] + coeffs[4][i])/5 for i in range(0,len(data.columns))]
    coeffs_std = [np.std([coeffs[0][i],coeffs[1][i],coeffs[2][i],coeffs[3][i],coeffs[4][i]]) for i in range(0,len(data.columns))]
    dfCoeffs = pd.DataFrame({'type':data.columns.values, 'coef':coeffs_avgd, 'std':coeffs_std})
    return dfCoeffs

def split(feat, data):
    X = data.drop([feat],1)
    y = data[feat]
    scaler = StandardScaler()
    X = X.as_matrix().astype(np.float)
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X,y, X_train, X_test, y_train, y_test

def linreg(X_train, X_test, y_train, y_test, data):
    # Create regression object
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    r2 = regr.score(X_test, y_test)
    regr.coef_
    coe = pd.DataFrame({'type':list(data.columns), 'coef':regr.coef_})

    img1 = StringIO.StringIO()
    img2 = StringIO.StringIO()
    plt.figure(2)
    plt.ylabel('')
    plt.xlabel('Predicted')
    plt.title('Linear Regression - Training Data')
    plt.scatter(regr.predict(X_train),y_train)
    plt.plot(y_train,y_train)
    plt.savefig(img1, format='png')
    img1.seek(1)
    trscat = base64.b64encode(img1.getvalue())

    plt.figure(3)
    plt.ylabel('')
    plt.xlabel('Predicted')
    plt.title('Linear Regression - Testing Data')
    plt.scatter(regr.predict(X_test),y_test,label='Actual')
    plt.plot(y_test,y_test)
    plt.savefig(img2, format='png')
    img2.seek(2)
    tescat = base64.b64encode(img2.getvalue())
    return r2, coe, trscat, tescat

def getDs(df):
    df_TWN = pd.get_dummies(df['TWN'])
    df_HDS = pd.get_dummies(df['HDS'])
    df_final = pd.concat([df[['YR','MSP','PWP','HIR','SCH','INF','WSC','NCR']],df_TWN,df_HDS],axis=1)
    return df_final

def sinCor(Type2, df2):
    df_cor = df2.corr()
    df_cor_s = df_cor[Type2]
    df_cor_s = df_cor_s.to_frame()
    table = df_cor_s.round(3)
    minval = table[Type2].min()
    minfeat = table.loc[table[Type2] == minval].index[0]
    tab = table[table[Type2] < 1]
    maxval = tab[Type2].max()
    maxfeat = tab.loc[tab[Type2] == maxval].index[0]
    return table, maxfeat, minfeat

def CorMat(table):
    minval = table.min()
    minfeat = []
    maxfeat = []
    ind = []
    for i in table.index:
        ind.append(i)
        minfeat.append(table.loc[table[i] == minval[i]].index[0])
        maxtab = table[table[i] < 1]
        maxv = maxtab[i].max()
        maxfeat.append(maxtab.loc[maxtab[i] == maxv].index[0])
    df11 = pd.DataFrame({'Category':ind,'Highest Direct':maxfeat,'Highest Inverse':minfeat})
    return df11

def corFeat(table):
    dfs = table.unstack()
    dfs = dfs.sort_values()
    minval = dfs.min()
    min1 = dfs.loc[dfs == minval].index[0]
    desa = df2.loc[df2['Category'] == min1[0],'Description'].iloc[0]
    desb = df2.loc[df2['Category'] == min1[1],'Description'].iloc[0]
    des2 = "%s and %s " %(desa,desb)
    dfs = dfs[dfs < 1]
    maxval = dfs.max()
    max1 = dfs.loc[dfs == maxval].index[0]
    desa = df2.loc[df2['Category'] == max1[0],'Description'].iloc[0]
    desb = df2.loc[df2['Category'] == max1[1],'Description'].iloc[0]
    des1 = "%s and %s " %(desa,desb)
    return max1, min1, des1, des2

def pricepred(town, infra, d2):
    data = d2[d2['YR'] == 2017]
    if infra == "I526":
        yr = 2020
        upd = "I-526 Extension to Johns and James Island"
        towns = ['CHS', 'KI', 'FOL']
        prox = .07
        inf = [.05,0,.1]
    elif infra == "NMS":
        yr = 2019
        upd = "North Meeting Street I-26 Exits"
        towns = ['CHS', 'NCH']
        prox = .02
        inf = [.02,.05]
    elif infra == "CR41":
        yr = 2020
        upd = "North Meeting Street I-26 Exits"
        towns = ['MTP', 'HAN', 'MNC']
        prox = .08
        inf = [.15,.05,.02]
    elif infra == "HW61":
        yr = 2025
        upd = "HWY 61 Widening"
        towns = ['NCH', 'SUM', 'GC']
        prox = .07
        inf = [.05,.05,.05]
    else:
        yr = 2018
        data['YR'] = yr
        fut = pd.concat([data,d2],axis=0)
        upd = "None"
        scaler = StandardScaler()
        fut = fut.as_matrix().astype(np.float)
        fut = scaler.fit_transform(fut)
        return fut, yr, upd
    data['YR'] = yr
    j = 0
    for t in towns:
        data.loc[data[t] == 1, 'PWP'] = data.loc[data[t] == 1, 'PWP'] * (1.0 - prox)
        data.loc[data[t] == 1, 'INF'] = data.loc[data[t] == 1, 'INF'] + inf[j]
        j = j + 1
    fut = pd.concat([data,d2],axis=0)
    scaler = StandardScaler()
    fut = fut.as_matrix().astype(np.float)
    fut = scaler.fit_transform(fut)
    return fut, yr, upd



@app.route('/')
def welcome():
    return render_template("welcome.html")

@app.route('/Data')
def DataSet():
    cat1 = df
    cat2 = df2
    return render_template("Data.html", tab1=cat1.to_html(), tab2=cat2.to_html())

@app.route('/Analysis_Home', methods = ['GET','POST'])
def AnHo():
    print("Starting Analysis")
    return render_template("Analysis_Home.html")

@app.route('/Analysis', methods = ['GET','POST'])
def analy():
    if request.method == 'POST':
        analy_type = request.form["analyType"]
        if analy_type == "Infrastructure":
            Type2 = "INF"
            state = "Kiawah is a gated island with infrastructure included in their private."
        elif analy_type == "Price":
            Type2 = "MSP"
            state = "Kiawah Island has the highest mean price."
        else:
            analy_type = "All Variables"
            Type2 = "All Variables"
            Desc = "Correllation Matrix:"

        if len(Type2) == 3:
            table, max1, min1 = sinCor(Type2, df)
            a1 = "The feature that has the largest direct corrrelation is: "
            Desc = df2.loc[df2['Category'] == Type2,'Description'].iloc[0]
            des1 = df2.loc[df2['Category'] == max1,'Description'].iloc[0]
            des2 = df2.loc[df2['Category'] == min1,'Description'].iloc[0]
            sans_ki = "We remove Kiawah Island because, "
            df_nki = df[df['TWN'] != 'KI']
            table2, max2, min2 = sinCor(Type2, df_nki)
            if max2 == "YR":
                des3 = "Year"
            else:
                des3 = df2.loc[df2['Category'] == max2,'Description'].iloc[0]
            des4 = df2.loc[df2['Category'] == min2,'Description'].iloc[0]
            a2 = "The feature that has the largest inverse corrrelation is: "
            cat = df2
            return render_template("Analysis.html", selection_a=analy_type, descr=Desc, selection_b=Type2, statement=sans_ki, statement2=state, Analysis1=a1, Analysis2=a2, Analysis3=a1, Analysis4=a2, feature1=max1, feature2=min1, descr1=des1, descr2=des2, descr3=des3, descr4=des4, feature3=max2, feature4=min2, tab=table.to_html(), tab2=table2.to_html(), cat=cat.to_html())
        else:
            cat = df2
            df_cor = df
            df_cor = df_cor.corr()
            table = df_cor.round(3)
            corrst = "The highest correlation factors by category are (Direct and Inverse):"
            table2 = CorMat(table)
            state = " "
            a1 = "The features that are most directly correlated are: "
            a2 = "The features that are most inversely correlated are: "
            max1,min1,des1,des2 = corFeat(table)
            max2 = table2['Highest Direct'].value_counts().idxmax()
            min2 = table2['Highest Inverse'].value_counts().idxmax()
            des3 = df2.loc[df2['Category'] == max2,'Description'].iloc[0]
            des4 = df2.loc[df2['Category'] == min2,'Description'].iloc[0]
            a3 = "The feature that most directly affects the dataset is "
            a4 = "The feature that has the most inverse affect on the dataset is: "
            return render_template("Analysis.html", selection_a=analy_type, descr=Desc, selection_b=Type2, tab=table.to_html(), statement=corrst, statement2=state, Analysis1=a1, Analysis2=a2, Analysis3=a3, Analysis4=a4, feature1=max1, feature2=min1, descr1=des1, descr2=des2, descr3=des3, descr4=des4, feature3=max2, feature4=min2, tab2=table2.to_html(), cat=cat.to_html())
    else:
        return render_template("Analysis_Home.html")

@app.route('/Prediction_Home', methods = ['GET','POST'])
def PreHo():
    print("Starting Prediction")
    return render_template("Prediction_Home.html")

@app.route('/Price_Pred', methods = ['GET','POST'])
def Pred():
    if request.method == 'POST':
        town = request.form["TWN"]
        infra = request.form["INF"]
        tns = df.loc[df['YR'] == 2017, 'TWN']
        tns = tns.values
        data = getDs(df)
        X, y, X_train, X_test, y_train, y_test = split('MSP', data)
        d2 = data.drop(['MSP'],1)
        r2_init, coe_i, trscat, tescat = linreg(X_train, X_test, y_train, y_test, d2)
        r2_sc, alpha_best, BR2 = bestR2(X, y, X_train, X_test, y_train, y_test)
        if town == "All" and infra == "NONE":
            fut = X
            yr = 2018
            upd = "None"
        else:
            fut, yr, upd  = pricepred(town, infra, d2)
        if r2_sc.max() > r2_init:
            regr = Lasso(alpha=alpha_best)
            rtype = "LASSO"
            regr.fit(X_train, y_train)
            y_pred = regr.predict(fut)
        else:
            regr = LinearRegression()
            rtype = "Linear Regression"
            regr.fit(X_train, y_train)
            y_pred = regr.predict(fut)
        pred = pd.DataFrame({'Town': tns, 'Pred':y_pred[:10]})
        if town != "All":
            cur = df_final.loc[(df_final['YR'] == 2017) & (df_final[town] == 1), 'MSP']
            cur = cur.values
            cur=cur[0]
            towndes = df2.loc[df2['Category'] == town,'Description'].iloc[0]
            twn_pred = pred.loc[pred['Town'] == town, 'Pred'].round(1)
            twn_pred = twn_pred.values
            twn_pred = twn_pred[0]
        else:
            cur = df_final.loc[df_final['YR'] == 2017, 'MSP']
            cur = cur.mean()
            towndes = "All Towns"
            twn_pred = np.average(y_pred[:10])
            twn_pred = np.round_(twn_pred,1)
        return render_template("Price_Pred.html", yr=yr, upd=upd, town=towndes, rtype=rtype, cur=cur, pred=twn_pred)
    else:
        return render_template("Prediction_Home.html")

@app.route('/Regression')
def Regress():
    data = getDs(df)
    X, y, X_train, X_test, y_train, y_test = split('MSP', data)
    d2 = data.drop(['MSP'],1)
    r2_init, coe_i, trscat, tescat = linreg(X_train, X_test, y_train, y_test, d2)
    if r2_init > .9:
        state = "The Linear Regression appears to be pretty good, now performing regularization to check further results using Lasso"
    else:
        state = "The Linear Regression is not very good, now perform regularization and Lasso to see if there is improvement"
    r2_sc, alpha_best, BR2 = bestR2(X, y, X_train, X_test, y_train, y_test)
    if r2_sc.max() > r2_init:
        state2 = "The Lasso (L1) method produces better results because the model is shrinking the less important coefficients to zero! Although we don't have a lot of coefficients, there are a few non-important ones"
    else:
        state2 = "The Lasso (L1) method is typically used for lots of features to shrinking the least important to essentially remove, however the features in this model are all relatively important."
    dfCoeffs = run_cv(d2,X,np.array(y),Lasso,alpha=alpha_best)
    grid, gbs, gbe = GSCV(X,y)
    if gbs > .9:
        result = "The grid score indicated that this is a good estimate for the model and will work well for predictions."
    else:
        result = "The grid score may be low, but we will use the model for predictions with some hesitancy on the results."
    return render_template("Regression.html", linreg_r2=r2_init, regul_r2=r2_sc.max(), linreg_co=coe_i.to_html(), linstate=state, regstate=state2, alpha=alpha_best, regcoe=dfCoeffs.to_html(), plot_url1=BR2, plot_url2=trscat, plot_url3=tescat, gridsc=gbs, gridest=gbe, result=result)


if __name__ == '__main__':
    app.run()





