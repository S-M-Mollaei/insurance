# main.py

from src.config import Config, init_environment
from src.loader import ARFFLoader
from src.preprocess import Preprocessor
from src.outlier import OutlierDetector
from src.dimensionality import PCAReducer
from src.analysis import CorrelationAnalyzer
from src.visualization import Visualizer
from src.anova import ANOVAAnalyzer, parse_variable_descriptions
from IPython.display import display
from src.data_prep import DataPreparator
from src.model import ModelTrainer



def main():
    # Step 1: Set up config and folders
    cfg = Config()
    paths = init_environment(cfg)

    # Step 2: Load and prepare all ARFF files
    loader = ARFFLoader(
        data_dir=paths["DATA_DIR"],
        cutoff_year=cfg.CUTOFF_YEAR,
        cutoff_quarter=cfg.CUTOFF_QUARTER
    )
    df_all = loader.load_all()

    # For now, just preview shape
    print(f"✅ Data loaded: {df_all.shape[0]} rows, {df_all.shape[1]} columns")

    # Step 3: Preprocess: clean + map sector names
    pre = Preprocessor(dim_dir=paths["DIM_DIR"], verbose=True)
    df_all = pre.remove_violations(df_all)
    df_all = pre.map_sector_names(df_all)
    feature_cols = pre.summarize(df_all)

    # Step 4: Outlier detection
    detector = OutlierDetector(feature_prefix="X", threshold=3.0, verbose=True)
    outlier_summary = detector.detect(df_all)

    # Step 5: PCA analysis
    pca_reducer = PCAReducer(n_components=2, feature_prefix="X", verbose=True, save_path=paths["FIG_DIR"])
    df_all_pca = pca_reducer.fit_transform(df_all)
    pca_reducer.plot(df_all_pca, hue="sector_name", definition="All Data") 

    # Filter PCA outliers
    df_filtered = pca_reducer.filter_outliers(df_all_pca, method="manual", limits=(-50, 50))
    pca_reducer.plot(df_filtered, hue="sector_name", definition="Manual Limits")

    df_filtered_iqr = pca_reducer.filter_outliers(df_all_pca, method="iqr")
    pca_reducer.plot(df_filtered_iqr, hue="sector_name", definition="IQR Limits")

    # Step 6: Correlation Analysis
    analyzer = CorrelationAnalyzer(feature_prefix="X", target_col="S", verbose=True, save_path=paths["FIG_DIR"])
    top_corr = analyzer.compute(df_all, top_n=10)
    analyzer.plot(top_corr)

    # Clean target column and save
    df_all = analyzer.clean_target(df_all, save_path= "./combined_data.csv")

    # Step 7: Visualizations
    viz = Visualizer(sector_map=pre.sector_map, verbose=True, save_path=paths["FIG_DIR"])
    viz.plot_sector_distribution(df_all, target_col="S")
    viz.plot_pre_post_covid(df_all, period_col="period")
    viz.plot_country_sector_heatmap(df_all, country_col="Country", sector_col="sector_name")

    # === Task One: ANOVA ===
    # Load description map
    desc_path = paths["DIM_DIR"] / "dimension.csv"
    description_map = parse_variable_descriptions(str(desc_path))

    # Instantiate analyzer
    anova = ANOVAAnalyzer(group_col="period", verbose=True, save_path=paths["FIG_DIR"])

    # --- 1. GLOBAL ANOVA ---
    anova_df_global = anova.run_global(df_all, feature_cols)

    # Select top features and plot
    top_global_features = anova_df_global.head(5)["feature"].tolist()
    anova.plot_global_top_features(df_all, top_global_features, description_map)

    # --- 2. SECTOR-WISE ANOVA ---
    anova_sector_df, sector_results = anova.run_by_sector(df_all, feature_cols)
    summary_df = anova.summarize_by_sector(sector_results)

    print("📈 Feature ranking by number of sectors with significant change:")
    display(anova_sector_df.head())
    print("\n📊 Summary of significant features across sectors:")
    display(summary_df.head())

    # Heatmap: sector × feature
    anova.pivot_heatmap(sector_results, df_all, description_map)

    # Per-sector boxplots
    anova.plot_by_sector_features(df_all, sector_results, description_map)


    # === Task Two: Data Preparation ===
    prep = DataPreparator(test_size=0.2, random_state=42, verbose=True)

    # Prepare features
    X_all, X_selected, y = prep.prepare_features(df_all, sector_results, target_col="S")

    # Split into train/test
    splits = prep.split(X_all, X_selected, y)

    # Unpack if needed
    X_train_all, X_test_all, y_train_all, y_test_all = splits["all"]
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = splits["selected"]

    #  Model Training & Evaluation 
    trainer = ModelTrainer(n_estimators=100, random_state=42, n_splits=5, verbose=True, path=paths["FIG_DIR"])

    # --- All features ---
    print("\n=== Training with ALL features ===")
    cv_all = trainer.cross_validate(X_train_all, y_train_all)
    clf_all = trainer.train_and_evaluate(X_train_all, X_test_all, y_train_all, y_test_all)
    trainer.plot_feature_importances(clf_all, X_all.columns, top_n=15)

    # --- Selected features ---
    print("\n=== Training with SELECTED features ===")
    cv_sel = trainer.cross_validate(X_train_sel, y_train_sel)
    clf_sel = trainer.train_and_evaluate(X_train_sel, X_test_sel, y_train_sel, y_test_sel)
    trainer.plot_feature_importances(clf_sel, X_selected.columns, top_n=15)

    # Hyperparameter Tuning with GridSearchCV
    print("\n=== Hyperparameter Tuning with GridSearchCV ===")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    print("\n=== Grid Search: ALL features ===")
    best_model_all = trainer.tune_hyperparameters(
        X_train_all, y_train_all, X_test_all, y_test_all,
        param_grid=param_grid, scoring="f1_macro"
    )

    print("\n=== Grid Search: SELECTED features ===")
    best_model_sel = trainer.tune_hyperparameters(
        X_train_sel, y_train_sel, X_test_sel, y_test_sel,
        param_grid=param_grid, scoring="f1_macro"
    )

if __name__ == "__main__":
    main()
