import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import PartialDependenceDisplay

def plot_shap_summary(model, X_test):
    """SHAP Summary Plotの作成"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    # LightGBMの場合、二値分類ならshap_values[1]が1の確率に対する寄与
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.show()

def plot_ice(model, X_test, features):
    """Individual Conditional Expectation (ICE) のプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model, X_test, features, 
        kind='both', # PDP (average) and ICE (individual)
        ax=ax
    )
    plt.title(f"ICE Plot for {features}")
    plt.tight_layout()
    plt.show()

def show_dice_counterfactuals(dice_exp, query_instance, total_CFs=3):
    """DiCEによる反実仮想例の表示"""
    # dice_exp: dice_ml.Explainer object
    # query_instance: 離脱しそうなサンプルのDataFrame
    dice_results = dice_exp.generate_counterfactuals(
        query_instance, total_CFs=total_CFs, desired_class="opposite"
    )
    dice_results.visualize_as_dataframe(show_only_changes=True)
