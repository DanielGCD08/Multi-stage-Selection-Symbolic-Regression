import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import platform
import numpy as np
from sympy import sympify,pretty,simplify
import sympy as sp
from pysr import PySRRegressor
from scipy.optimize import curve_fit
import joblib
ERRORVALUE=1e12
niterations=100
populations=50 
nodemaxsize=40 
min_complexity=7
max_complexity=25
top_n=10
procs=7 
maxfev=10000
#alleqcsvfile = "alleq.csv"
datapath = "d:/daniel/danbaishiyan/danbai91"

eqpath="d:/daniel/danbaishiyan"
#alleqcsvfile =os.path.join(eqpath,"data_95_eq.csv")
alleqcsvfile =os.path.join(eqpath,
f"lc912_alleq_ni{niterations}_po{populations}_maxnode{nodemaxsize}_sin_sqrt_0.csv")

unary_operators=["cos","sin","exp","sqrt"]
def read_data(filefullpath):
    #print("Reading data from:", filefullpath)
    data = pd.read_csv(filefullpath,names = ["xaxis0","yaxis"])  
    time = np.array(list(map(float,data["xaxis0"][1:])))
    time = time - time[0]
    
    y = np.array(list(map(float,data["yaxis"][1:])))
    #sub_df = data.iloc[1000:4000]
    #sub_df.to_csv(os.path.join(path,"cut_"+name))
    time= np.array(time, dtype=float)
    y = np.array(y, dtype=float)
    time = time.reshape(-1,1)

    return time,y
def symbolic_regression(x,y):
    model = PySRRegressor(
    niterations=niterations,        
    binary_operators=["+", "-", "*", "/","^"], 
    unary_operators=unary_operators, 
    

    
    populations=populations,         
    maxsize=nodemaxsize,             

    complexity_of_operators={"cos":3,"sin":3,"^": 3, "exp": 3,"sqrt": 2}, 
  
    model_selection="best",

    parsimony=0.001,
    should_optimize_constants=True,   
    batching=True,                
    batch_size=10000,              
    warm_start=True,
    procs=procs, 
    verbosity=1
    )
    return model.fit(x,y)

def select_top_pysr_models(equations_df, min_complexity=min_complexity, max_complexity=max_complexity, top_n=top_n):

    filtered = equations_df[
        (equations_df['complexity'] >= min_complexity) &
        (equations_df['complexity'] <= max_complexity)
    ]
    top_models = filtered.nsmallest(top_n, 'loss')
    return top_models.reset_index(drop=True)
def para_processing(eqfile):
    df_allequ = pd.read_csv(eqfile)   
    for idx,row in df_allequ.iterrows():
        para_eq, para_list,orgpara_list = replace_floats_toConstant(sp.sympify(row["eq"]))
        df_allequ.loc[idx, "para_eq"] = str(para_eq)
        df_allequ.loc[idx, "para_list"] = str(para_list)
        df_allequ.loc[idx, "orgpara_list"] = str(orgpara_list)
    df_allequ.to_csv(eqfile, index=True)

def FirstSelect(filelist):
    df_allequ = pd.DataFrame(columns=['orgfile', 'complexity', 'loss','score','eq','orgpara_list','para_eq','para_list','R2_loss','eva_loss','redundant'])
    for file in filelist:
        print("pysr:",file)
        x,y=read_data(file)
        model = symbolic_regression(x,y)
        #print("best equ",model.get_best())
        top10 = select_top_pysr_models(model.equations_)
        #print("top10",(top10))
        for index, row in top10.iterrows():
            para_eq, para_list,const_list = replace_floats_toConstant(sp.sympify(row["equation"]))
            newrow=pd.DataFrame([{"orgfile":os.path.basename(file),'complexity':row["complexity"],'loss':row['loss'],
                    'score':row['score'],'eq':row['sympy_format'],'orgpara_list':str(const_list),'para_eq':str(para_eq),'para_list':str(para_list),
                    'R2_loss':0,'eva_loss':0,'redundant':0 }])
            df_allequ = pd.concat([df_allequ, newrow], ignore_index=True)

        df_allequ.to_csv(alleqcsvfile, index=True)

def replace_floats_toConstant(expr):
    float_to_symbol = {}
    param_list = []
    const_ = [] 
    counter = 0
    '''def replace_floats(expr):
        nonlocal counter
        if expr.is_Number:
            if expr not in float_to_symbol:
                param_name = f"p{counter}"
                param_sym = sp.Symbol(param_name)
                float_to_symbol[expr] = param_sym
                param_list.append(param_sym)
                counter += 1
            return float_to_symbol[expr]
        elif expr.args:
            return expr.func(*[replace_floats(arg) for arg in expr.args])
        else:
            return expr
    '''
    def replace_floats_toConstant(expr):
        nonlocal counter
        if expr.is_Number:
            if expr not in float_to_symbol:
                param_name = f"p{counter}"
                param_sym = sp.Symbol(param_name)
                float_to_symbol[expr] = param_sym
                param_list.append(param_sym)
                const_.append(expr)  
                counter += 1
            return float_to_symbol[expr]
        elif expr.args:
            return expr.func(*[replace_floats_toConstant(arg) for arg in expr.args])
        else:
            return expr

    new_expr = replace_floats_toConstant(expr)
    return new_expr, param_list, const_
    #return replace_floats(expr),param_list

def make_wrapper(X,f):
    def wrapper(Xdata, *params):
        args = list(params) + [Xdata[:, i] for i in range(Xdata.shape[1])]
        with np.errstate(all="ignore"):
            y_pred = f(*args)
        if np.any(~np.isfinite(y_pred)):
              return np.full_like(y_pred, ERRORVALUE)
        if np.any(~np.isfinite(y_pred)):
              return np.full_like(y_pred, ERRORVALUE)  # 返回大残差
        return y_pred 
    return wrapper

def evalateAllEq(eqfile,datafilelist):
    df_allequ = pd.read_csv(eqfile)
    rows=[]
    for idx,row in df_allequ.iterrows():
         eqn_symbolic = sp.sympify(row["para_eq"])
         param_list = [sp.symbols(i) for i in row["para_list"][1:-1].split(",")]
         orgpara_list= [float(sp.sympify(i)) for i in row["orgpara_list"][1:-1].split(",")]
         
         #eqn_symbolic, param_list = replace_floats_toConstant(sp.sympify(row["eq"]))
         #df_allequ.loc[idx, "cons_eq"] = str(eqn_symbolic)
         print("Symbolic formula with parameters:", eqn_symbolic)
         #print("Detected parameters:", param_list)
         x_symbols = [s for s in eqn_symbolic.free_symbols if str(s).startswith("x")]
         f = sp.lambdify(param_list + x_symbols, eqn_symbolic, "numpy")
         x,y=read_data(datafilelist[0])
         wrapper = make_wrapper(x,f)
         loss = []
         R2loss = []
         for file in datafilelist:
             x,y=read_data(file)
             try:
                popt, _ = curve_fit(wrapper, x, y, p0=orgpara_list,maxfev=maxfev)
                residuals = y - wrapper(x, *popt)
                mse = np.mean(residuals**2)
                SS_res = np.sum(residuals**2)
                SS_tot = np.sum((y - np.mean(y))**2)
                R2 = 1 - (SS_res / SS_tot)
             except RuntimeError as e:
               print("fit error:", e)
               popt = None  
               mse = ERRORVALUE
               R2 = ERRORVALUE


             if not np.isfinite(mse):
                  mse = ERRORVALUE 
             if np.isnan(mse):
                 mse = ERRORVALUE
             if not np.isfinite(R2):
                  R2 = ERRORVALUE
             if np.isnan(R2):
                 R2 = ERRORVALUE
             #print("mseloss:",mse)
             loss.append(mse)
             R2loss.append(R2)

         meanloss = np.mean(np.array(loss))
         meanR2 = np.mean(np.array(R2loss))
         if not np.isfinite(meanloss):
              meanloss = ERRORVALUE
         if np.isnan(meanloss):
              meanloss = ERRORVALUE
         if not np.isfinite(meanR2):
              meanR2 = ERRORVALUE
         if np.isnan(meanR2):
              meanR2 = ERRORVALUE
         print(meanloss)
         df_allequ["eva_loss"] = df_allequ["eva_loss"].astype(float)
         df_allequ["R2_loss"] = df_allequ["R2_loss"].astype(float)  

         df_allequ.loc[idx, "eva_loss"] = float(meanloss)
         df_allequ.loc[idx, "R2_loss"] = float(meanR2)
         df_allequ.to_csv(eqfile, index=True)

def compare_expr(expr1, expr2, symbols, trials=5, tol=1e-6):

    diff = sp.simplify(expr1 - expr2)
    if diff == 0:
        return True

    eq = expr1.equals(expr2)
    if eq is True:
        return True
   
    for _ in range(trials):
        subs_dict = {}
        for sym in symbols:
            val = np.random.uniform(1, 5)  # 避免除以0
            subs_dict[sym] = val
        try:
            v1 = float(expr1.subs(subs_dict))
            v2 = float(expr2.subs(subs_dict))
            if abs(v1 - v2) > tol:
                return False
        except Exception:
            continue 
    
    return True

def remove_redundant_equations(eqfile):

    df_allequ = pd.read_csv(eqfile)
    for idx,row in df_allequ.iterrows():
         eqn_symbolic = sp.sympify(row["para_eq"])
         free_symbols = list(eqn_symbolic.free_symbols)
         if idx % 10 == 0:
             print(f"remove redundant:{idx}/{len(df_allequ)}")
         for idx_i,row_i in df_allequ.iterrows():
             if idx_i<= idx:
                 continue
             eqn_symbolic_i = sp.sympify(row_i["para_eq"])
             #print(f"  compare {idx} with {idx_i},",row["para_eq"],row_i["para_eq"])
             if len(row["para_eq"])==len(row_i["para_eq"]) and len(row["para_list"])==len(row_i["para_list"]): 
                if compare_expr(eqn_symbolic, eqn_symbolic_i, free_symbols):
                   df_allequ.loc[idx, "redundant"] = idx_i
    df_allequ.to_csv(eqfile, index=True)

def get_file_paths_bak(directory):
    file_paths = []
    print("Walking through directory:", directory)
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths
def get_file_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
         
        
if __name__=="__main__":
     filelist = get_file_paths(directory = datapath)
     #print(filelist)
     FirstSelect(filelist)
     #para_processing(alleqcsvfile)

     remove_redundant_equations(alleqcsvfile)
     evalateAllEq(alleqcsvfile,filelist)
     print("success!")
 


