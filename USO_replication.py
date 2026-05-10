######################Pyderivatives is required to run this replication folder. Install the package at https://github.com/Julian-Beatty/PyDerivatives_replication. 
#A guide for installation can be found at the github, or in the readme. 
####################


from pyderivatives import *
import importlib.resources as ir
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
########################################Change Working Directory to Replication Folder
##Simple check

# Get current working directory
cwd = Path.cwd()

# Check if current folder name is exactly "Replication Folder"
if cwd.name == "Replication Folder":
    print("Working directory is Replication Folder")
else:
    print(f"Current working directory is: {cwd}")
    raise ValueError("Working directory is not 'Replication Folder'. Change working directory to Replication Folder where required and preloaded data exist.")



###################################################################Part 0 Data Processing#########################################
###########Load Treasury data

yield_curve_files=["required_data/daily-treasury-rates (1).csv",
"required_data/par-yield-curve-rates-1990-2022.csv",
"required_data/daily-treasury-rates (3).csv",
"required_data/daily-treasury-rates (2).csv"]


##########Setting up Yield Curve fitting to first 5 years.
df=build_yield_dataframe(yield_curve_files)
rc_object=create_yield_curve(df)
sve_nsurface=rc_object.fit("svensson",grid_days=[1,365*3],fit_days_window=[1,365*5])



###########Initialize Option Market class 
option_market_class=OptionMarketStandardizer(option_data_filename_prefix="full_uso_options.csv",stock_data_filename="USO_stock.csv",
                         rate_curve_df=sve_nsurface, 
                         data_directory_path="required_data",
                         vendor_name="optionmetrics",
                         stock_date_col="date",
                         stock_price_col="price",
                         rate_date_col="Date",
                         ticker="USO")





###########Data Preprocessing
otm_calls = option_market_class.keep_options(
    maturity_filter=[7, 60],
    moneyness_filter=[1, 1.3],
    min_volume_filter=-1.0,
    min_price_filter=0.05,
    option_right_filter="c",
    date_filter=["2008-01-01","2026-12-31"]
    
)

otm_puts = option_market_class.keep_options(
    maturity_filter=[7, 60],
    moneyness_filter=[0.7, 0.99],
    min_volume_filter=-1.0,
    min_price_filter=0.05,
    option_right_filter="p",
    date_filter=["2008-01-01","2026-12-31"]
)



otm_puts_tocalls=put_call_parity(otm_puts)
OTM_options_only_df=pd.concat([otm_calls,otm_puts_tocalls]).sort_values(["date","exdate","strike"]).reset_index(drop=True)


x0 = dict(

    # fast variance factor (front-end skew / short maturities)
    v0=0.04,    # initial variance v1(0)
    theta=0.5,  # long-run mean of v1
    kappa=6.0, # mean reversion speed of v1 (large => fast)
    sigma_v=0.2,  # vol-of-vol of v1 (controls smile strength)
    rho=-0.6,   # corr(return shock, v1 shock): negative => left skew



    # Kou jumps (double exponential jump sizes)
    lam=0.6,     # jump intensity: expected jumps per year
    p_up=0.50,   # P(jump is upward)
    eta1=20.0,   # upward jump rate: E[J | J>0] = 1/eta1
    eta2=20.0,   # downward jump rate: E[-J | J<0] = 1/eta2
)


############Setting up boundaries and initial conditions for Heston-Kou model
lb = dict(
v0=0.005, theta=0.4, kappa=0.5,  sigma_v=0.1, rho=-0.85,
lam=0.02, p_up=0.05, eta1=10.0, eta2=10.0,
)

ub = dict(
v0=5.0, theta=1.0, kappa=5000.0, sigma_v=5.0, rho=0.85,
lam=30.0, p_up=0.95, eta1=20.0, eta2=20.0,
)


###################################################################Part 1 data calibration#########################################
base_dir = Path.cwd()
multisurface_dir = base_dir / "results_tables_and_figures/gallery/multisurface"
multisurface_dir.mkdir(parents=True, exist_ok=True)


USO_RND_dict_2008_2025={}
date_list=list(OTM_options_only_df["date"].unique())
for date in date_list:
    
    
    ####Select a call-surface Slice and begin arbitrage Repair from Cohen 2020
    option_day_df=OTM_options_only_df[OTM_options_only_df["date"]==date]
    readable_date = pd.to_datetime(date).strftime("%Y_%m_%d")
    formatted_date = pd.to_datetime(date).strftime("%Y-%m-%d")
    out = CallSurfaceArbRepair(RepairConfig()).repair_one_date(option_day_df)
    pdict = out["plot_data"]
    repaired_df=out["df_rep"]
    
    ##Optional Diagnostic Plots:
    fg=plot_perturb(pdict,save=f"results_tables_and_figures/gallery/arbitrage_heatmaps/arbitrage_heatmap_{formatted_date}.png") ## Perbutation repair heatmap
    
    ######## Prepare for calibration
    day=make_day_from_df(repaired_df,price_col="C_rep")

    
    
    ###Optional Calibration Hyperparameters
    safety_clip=SafetyClipConfig(enabled=True, clip_left=True,clip_right=True,center="mode") ##Clips the RND for numerical stability if needed
    iv_cfg = IVConfig(
        sigma_init=0.3,         # starting guess for Newton
        sigma_lo=1e-8,          # lower bracket / floor
        sigma_hi=5.0,           # upper bracket (increase if you get "no root" for deep OTM)
        newton_max_iter=150,     # Newton iterations before fallback
        newton_tol=1e-11,        # convergence tolerance
        vega_floor=1e-11,       # treat vega below this as numerically unreliable
        brent_maxiter=150,      # if Newton fails, Brent iterations
        time_value_floor=1e-11,  # guards: if call time value too tiny, skip/return near 0 IV
        reject_low_vega=1e-11,   # optionally reject points with too-small vega
    )
    ###### Select model and enter bounds
    
    pr = GlobalSurfacePricer("heston_kou",Umax=1500.0, n_quad=1500)
    pr.fit(day,x0=x0,bounds=(lb,ub))
    T_grid=np.arange(7,61)/365 #Evalute the surface each day from day 7 to 2 months.
    
    ###Price options on standardized grid
    results = pr.price(day,
                     safety_clip=safety_clip,
                     iv_cfg=iv_cfg,
                     grid_mode="moneyness", 
                     m_bounds=(0.1, 3.5),
                     m_grid_n=600, 
                     T_grid=T_grid,
                     compute_moments=True,compute_iv=True,compute_rnd=True,compute_delta=False,compute_cdf=True)
    print(results["params"])

    
    
    #######Plot Option data
    USO_RND_dict_2008_2025[date]=results

    
    fig=panels.call_panels(results,day=day,n_panels=6,title=f"Call Curves for USO on {formatted_date}",K_pad_abs=5,
                       save=f"results_tables_and_figures/gallery/fitted_calls/fittedcalls_{readable_date}.png")
    
    fig=panels.rnd_panels(results, n_panels=5,panel_shape=(5,1), title=f"USO: RNDs on {formatted_date}", x_axis="r",x_bounds=(-2.0,1.5),
                          save=f"results_tables_and_figures/gallery/rnds/rnd_{readable_date}.png")

    
    # # ##Surface plots
    
    fig=surfaces.call_surface_vs_observed(results,day=day,title=f"USO: Fitted Call Surface on {formatted_date}",
                                          save=f"results_tables_and_figures/gallery/call_surface/callsurface_{readable_date}.png")
    


    fig=surfaces.rnd_surface_plot(results,title=f"USO: Risk Neutral Surface {formatted_date}",cmap="viridis", interactive=False, x_axis="r", x_bounds=(-0.5,0.5),
                                  save=f"results_tables_and_figures/gallery/RNDsurface/rnd_surface_{readable_date}.png")
    
    fig=surfaces.iv_surface_plot(
        results,
        title=f"USO: IV Surface on {formatted_date}",
        cmap="inferno",
        save=f"results_tables_and_figures/gallery/IV_surface/ivsurface_{readable_date}.png",
        interactive=False
    )
    
    fig = plt.figure(figsize=(20, 7))
    
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    
    _, ax1, surf1 = surfaces.call_surface_vs_observed(
        results,
        day=day,
        ax=ax1,
        title="Fitted Call Surface",
        show=False,
        x_axis="r",
        x_bounds=(-0.30, 0.30),
    )
    
    _, ax2, surf2 = surfaces.iv_surface_plot(
        results,
        ax=ax2,
        x_axis="r",
        x_bounds=(-0.30, 0.30),
        title=f"IV Surface",
        cmap="inferno",
        interactive=False,
        show=False,
    )
    
    _, ax3, surf3 = surfaces.rnd_surface_plot(
        results,
        ax=ax3,
        title=f"Risk Neutral Surface",
        cmap="viridis",
        x_axis="r",
        x_bounds=(-0.30, 0.30),
        interactive=False,
        show=False,
    )
    
    fig.suptitle(f"USO Option Implied Surfaces on {formatted_date}", fontsize=20, y=0.95)
    fig.savefig(
         f"results_tables_and_figures/gallery/multisurface/multi_surface_{readable_date}.png",
         dpi=300
    )
    
    plt.show()
    


##############################################Post Diagnostics    of RND calibration ###########################


with open("preloaded_data/USO_RND_2008_2025.pkl", "rb") as f:
    pickle_2008_2025 = pickle.load(f)
    
    
USO_RND_dict_2008_2025=pickle_2008_2025["result_dict"]
stock_df=pickle_2008_2025["stock_df"]


########################################### Split RND dictionaries into subsamples 
##If you did not preload the RNDs using the above code block you must run this code block.
from pyderivatives.Useful_functions.merging_helpers import slice_dict_by_date
RND_dict_2008_2019 = slice_dict_by_date(
    USO_RND_dict_2008_2025,
    "2008-01-01",
    "2019-12-31"
)

RND_dict_oil_crisis = slice_dict_by_date(
    USO_RND_dict_2008_2025,
    "2014-06-01",
    "2016-01-31"
)

RND_dict_normal_part_1 = slice_dict_by_date(
    USO_RND_dict_2008_2025,
    "2008-01-01",
    "2014-05-31"
)

RND_dict_normal_part_2 = slice_dict_by_date(
    USO_RND_dict_2008_2025,
    "2016-02-01",
    "2019-12-31"
)

RND_dict_Normal= {**RND_dict_normal_part_1, **RND_dict_normal_part_2}


RND_dict_2013_2025 = slice_dict_by_date(
    USO_RND_dict_2008_2025,
    "2013-09-25",
    "2025-08-29"
)


#####################################################Part 2 Estimation of Physical Densities and Pricing kernels
####This section estimates the physical densities and pricing kernels of USO. The requirements are the RND density dictionaries which can be autoloaded for speed. Otheriwse
#skip this section
############################################# Load RND dictionaries ##################################
    

############################################
ts_df, summary_df, latex = call_fit_error_timeseries(
    USO_RND_dict_2008_2025,
    metrics=("rmse", "mape"),
    plot=True,
    latex_caption="Daily call-surface calibration errors.",
    latex_label="tab:daily_call_calib_errors",
    latex_out="results_tables_and_figures/calibration_tables/table_fitted_call_errors.tex",
    decimals=4,
    include_percent_for=("mape",),   # will print MAPE * 100
    clip_at_p95=True,
    save_plots=True,
    plot_dir="results_tables_and_figures/calibration_tables",

)




####################Estimating the Pricing Kernel parameters###########################
###(1) Original Sample 2008 jan 1 - 2019 December 31
###(2) Crisis Sample: 2014 June 1  - 2016 Jan 31 
###(3) Normal Sample: 2008 jan 1 - 2014 June 1  | 2014 June 2 - 2019 December 31
###(4) Extended Sample: 2013 September 26 - 2025 December 31
logreturns=compute_horizon_returns_backward(
    stock_df,
    horizon=1,
    group_col=None
)


#This part may take some time.
pk_fit_2008_2019_N2K2 = estimate_pricing_kernel_global(
        RND_dict_2008_2019,
        stock_df,
        spot_col="price",
        dataset_tag="USO_2008_2019_N2K2",
        theta_spec=ThetaSpec(N=2, Ksig=2),
        bootstrap=BootstrapSpec(enabled=False),
        cache=CacheSpec(use_disk=True, folder="pk_cache"), ###Use_disk=True uses the cached parameters to run code quickly.
        maxiter=400,
        min_obs_per_T=5,
        eval_spec=EvalSpec.from_R_bounds((0.10, 2.5), r_grid_size=350),
        rnd_tail_alpha=(0.05, 0.95),
        diag_build_obs=True,
    )    
pk_fit_crisis_N2K2 = estimate_pricing_kernel_global(
        RND_dict_oil_crisis,
        stock_df,
        spot_col="price",
        dataset_tag="USO_crisis_N2K2",
        theta_spec=ThetaSpec(N=2, Ksig=2),
        bootstrap=BootstrapSpec(enabled=False),
        cache=CacheSpec(use_disk=True, folder="pk_cache"),
        maxiter=400,
        min_obs_per_T=5,
        eval_spec=EvalSpec.from_R_bounds((0.10, 2.5), r_grid_size=350),
        rnd_tail_alpha=(0.05, 0.95),
        diag_build_obs=True,
    )    

pk_fit_normal_N2K2 = estimate_pricing_kernel_global(
        RND_dict_Normal,
        stock_df,
        spot_col="price",
        dataset_tag="USO_normal_N2K2",
        theta_spec=ThetaSpec(N=2, Ksig=2),
        bootstrap=BootstrapSpec(enabled=False),
        cache=CacheSpec(use_disk=True, folder="pk_cache"),
        maxiter=400,
        min_obs_per_T=5,
        eval_spec=EvalSpec.from_R_bounds((0.10, 2.5), r_grid_size=350),
        rnd_tail_alpha=(0.05, 0.95),
        diag_build_obs=True,
    )    



pk_fit_2013_2025_N2K2 = estimate_pricing_kernel_global(
        RND_dict_2013_2025,
        stock_df,
        spot_col="price",
        dataset_tag="USO_2013_2025_N2K2",
        theta_spec=ThetaSpec(N=2, Ksig=2),
        bootstrap=BootstrapSpec(enabled=False),
        cache=CacheSpec(use_disk=True, folder="pk_cache"),
        maxiter=400,
        min_obs_per_T=5,
        eval_spec=EvalSpec.from_R_bounds((0.10, 2.5), r_grid_size=350),
        rnd_tail_alpha=(0.05, 0.95),
        diag_build_obs=True,
    )  

    


##################Estimating Pricing Kernel Surface and Physical densitiees
asset_name = "OIL"

physical_dict_2013_2025_N2K2 = {}
physical_dict_2008_2019_N2K2 = {}
physical_dict_normal_N2K2 = {}
physical_dict_crisis_N2K2 = {}

dates_2013_2025 = set(RND_dict_2013_2025.keys())
dates_2008_2019 = set(RND_dict_2008_2019.keys())
dates_normal    = set(RND_dict_Normal.keys())
dates_crisis    = set(RND_dict_oil_crisis.keys())

# If you want one master date list, pick union or intersection depending on intent:
date_list = sorted(dates_2013_2025 | dates_2008_2019 | dates_normal | dates_crisis)

def _ensure_parent(save_path: str | Path) -> None:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

for date in date_list:
    readable_date = pd.to_datetime(date).strftime("%Y_%m_%d")
    formatted_date = pd.to_datetime(date).strftime("%Y-%m-%d")


    # -------- 2013-2025 --------
    if date in RND_dict_2013_2025:
        out_2013_2025 = evaluate_anchor_surfaces_with_theta_master(
            RND_dict_2013_2025[date],
            theta_master=pk_fit_2013_2025_N2K2["theta_master"],
            theta_spec=ThetaSpec(**pk_fit_2013_2025_N2K2["theta_spec"]),
            eval_spec=EvalSpec(**pk_fit_2013_2025_N2K2["eval_spec"]),
            safety_clip=SafetyClipSpec(enabled=True, floor=0.0),
        )
        out_2013_2025["pk_fit"] = pk_fit_2013_2025_N2K2
        physical_dict_2013_2025_N2K2[date] = out_2013_2025

        save_path = f"results_tables_and_figures/gallery/pricing_kernel_2013_2025/{asset_name}_pricing_kernel_{readable_date}.png"
        _ensure_parent(save_path)

        v=P_Q_K_multipanel(
            out_2013_2025,
            trunc_mode="cdf+rbounds",
            r_bounds=(-0.9, 0.9),
            kernel_yscale="linear",
            panel_shape=(2, 4),
            title=f"{asset_name}: Pricing kernel vs Q vs P on {formatted_date}",
            save=save_path,
            ptail_alpha=(0.01, 0.01),
            x_space="log",  # IMPORTANT because r_bounds includes negatives
        )

    # -------- 2008-2019 --------
    if date in RND_dict_2008_2019:
        out_2008_2019 = evaluate_anchor_surfaces_with_theta_master(
            RND_dict_2008_2019[date],
            theta_master=pk_fit_2008_2019_N2K2["theta_master"],
            theta_spec=ThetaSpec(**pk_fit_2008_2019_N2K2["theta_spec"]),
            eval_spec=EvalSpec(**pk_fit_2008_2019_N2K2["eval_spec"]),
            safety_clip=SafetyClipSpec(enabled=True, floor=0.0),
        )
        out_2008_2019["pk_fit"] = pk_fit_2008_2019_N2K2
        physical_dict_2008_2019_N2K2[date] = out_2008_2019

        save_path = f"results_tables_and_figures/gallery/pricing_kernel_2008_2019/{asset_name}_pricing_kernel_{readable_date}.png"
        _ensure_parent(save_path)

        # P_Q_K_multipanel(
        #     out_2008_2019,
        #     trunc_mode="cdf+rbounds",
        #     r_bounds=(-0.6, 0.6),
        #     kernel_yscale="linear",
        #     panel_shape=(2, 4),
        #     title=f"{asset_name}: Pricing kernel vs Q vs P on {formatted_date}",
        #     save=save_path,
        #     ptail_alpha=(0.01, 0.01),
        #     x_space="log",
        # )

    # -------- Normal --------
    if date in RND_dict_Normal:
        out_normal = evaluate_anchor_surfaces_with_theta_master(
            RND_dict_Normal[date],
            theta_master=pk_fit_normal_N2K2["theta_master"],
            theta_spec=ThetaSpec(**pk_fit_normal_N2K2["theta_spec"]),
            eval_spec=EvalSpec(**pk_fit_normal_N2K2["eval_spec"]),
            safety_clip=SafetyClipSpec(enabled=True, floor=0.0),
        )
        out_normal["pk_fit"] = pk_fit_normal_N2K2
        physical_dict_normal_N2K2[date] = out_normal

        save_path = f"results_tables_and_figures/gallery/pricing_kernel_normal/{asset_name}_pricing_kernel_{readable_date}.png"
        _ensure_parent(save_path)

        # P_Q_K_multipanel(
        #     out_normal,
        #     trunc_mode="cdf+rbounds",
        #     r_bounds=(-0.6, 0.6),
        #     kernel_yscale="linear",
        #     panel_shape=(2, 4),
        #     title=f"{asset_name}: Pricing kernel vs Q vs P on {readable_date}",
        #     save=save_path,
        #     ptail_alpha=(0.01, 0.01),
        #     x_space="log",
        # )

    # -------- Crisis --------
    if date in RND_dict_oil_crisis:
        out_crisis = evaluate_anchor_surfaces_with_theta_master(
            RND_dict_oil_crisis[date],
            theta_master=pk_fit_crisis_N2K2["theta_master"],
            theta_spec=ThetaSpec(**pk_fit_crisis_N2K2["theta_spec"]),
            eval_spec=EvalSpec(**pk_fit_crisis_N2K2["eval_spec"]),
            safety_clip=SafetyClipSpec(enabled=True, floor=0.0),
        )
        out_crisis["pk_fit"] = pk_fit_crisis_N2K2
        physical_dict_crisis_N2K2[date] = out_crisis

        save_path = f"results_tables_and_figures/gallery/pricing_kernel_crisis/{asset_name}_pricing_kernel_{readable_date}.png"
        _ensure_parent(save_path)

        # P_Q_K_multipanel(
        #     out_crisis,
        #     trunc_mode="cdf+rbounds",
        #     r_bounds=(-0.6, 0.6),
        #     kernel_yscale="linear",
        #     panel_shape=(2, 4),
        #     title=f"{asset_name}: Pricing kernel vs Q vs P on {readable_date}",
        #     save=save_path,
        #     ptail_alpha=(0.01, 0.01),
        #     x_space="log",
        # )

####################################Ploting Pricing Kernel evolution over select maturities
base_dir = Path.cwd()
pricing_kernel_dir = base_dir / "results_tables_and_figures/figures"
pricing_kernel_dir.mkdir(parents=True, exist_ok=True)


fig = plt.figure(figsize=(15, 15))

axes = [
    fig.add_subplot(221, projection="3d"),
    fig.add_subplot(222, projection="3d"),
    fig.add_subplot(223, projection="3d"),
    fig.add_subplot(224, projection="3d"),
]

T_list = [7/365, 14/365, 21/365, 28/365]
titles = ["7 Days", "14 Days", "21 Days", "28 Days"]

for ax, T, ttl in zip(axes, T_list, titles):
    _, ax, surf = plot_pricing_kernel_3d_surface_by_T(
        physical_dict_2013_2025_N2K2,
        T_target=T,
        ax=ax,
        fP_key = 'fP_r_surface',
        x_axis="r",
        x_bounds=(-0.25, 0.25),
        interactive=False,
        show=False,
        title=f"T = {ttl}",
        add_colorbar=False,
        stride=60,
    )

fig.suptitle("Pricing Kernel Surfaces Across Maturities", fontsize=18)

plt.subplots_adjust(
    left=0.03,
    right=0.95,
    bottom=0.05,
    top=0.92,
    wspace=0.15,
    hspace=0.25
)

# ✅ SAVE HERE
fig.savefig(
    f"results_tables_and_figures/figures/Pricing Kernel Surface Across Maturities.png",
    dpi=300,
    pad_inches=0.3   # 🔥 prevents clipping (important for 3D)
)

plt.show()

###############################################Part 3 Statistical Analysis with Quantil Regressions
moments_2008_2019_dict = extract_moment_premia_timeseries(
    physical_dict=physical_dict_2008_2019_N2K2,
    rnd_dict=RND_dict_2008_2019,
    moments=("vol_ann", "skew", "kurt"),
    T_days=[7,14,21,28,35,60],
)

moments_crisis_dict = extract_moment_premia_timeseries(
    physical_dict=physical_dict_crisis_N2K2,
    rnd_dict=RND_dict_oil_crisis,
    moments=("vol_ann", "skew", "kurt"),
    T_days=[7,14,21,28,35,60],
)

moments_normal_dict = extract_moment_premia_timeseries(
    physical_dict=physical_dict_normal_N2K2,
    rnd_dict=RND_dict_Normal,
    moments=("vol_ann", "skew", "kurt"),
    T_days=[7,14,21,28,35,60],
)

moments_2013_2025_dict = extract_moment_premia_timeseries(
    physical_dict=physical_dict_2013_2025_N2K2,
    rnd_dict=RND_dict_2013_2025,
    moments=("vol_ann", "skew", "kurt"),
    T_days=[7,14,21,28,35,60],
)



dictionaries_to_summarize={"USO":moments_2008_2019_dict}
tabs = summary_stat(
    dictionaries_to_summarize,
    horizons=[7,14,21,28,35,60],
    which=("rnd",),   
    digits=2,
    save_dir="results_tables_and_figures/summary_tables",
    file_stub="table_summary_stats_RND"
)

plot_return_and_option_moments(
    logreturns,
    moments_2008_2019_dict,
    horizons=[21],
    moment_type="both",
    truncate_to_moments=True,
    truncate_mode="intersection",
    save=f"results_tables_and_figures/figures/plot_moment_time_series.png"
)


investment_horizon=[7,14,21,28,35,60]

taus = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
)

segments = {
    "low":  taus[0:6],
    "mid":  taus[6:13],
    "high": taus[13:19],
}
assymetric_regression_2008_2019_results={}
assymetric_regression_crisis_results={}
assymetric_regression_normal_results={}
assymetric_regression_2013_2025_results={}
assymetric_regression_2013_2025rnd_results={}


##################################################################Pre Load Quantile Regression Results####################################

with open("preloaded_data/USO_2008_2019_results.pkl", "rb") as f:
    assymetric_regression_2008_2019_results = pickle.load(f)
    
with open("preloaded_data/USO_crisis_results.pkl", "rb") as f:
    assymetric_regression_crisis_results = pickle.load(f)
    
with open("preloaded_data/USO_normal_results.pkl", "rb") as f:
    assymetric_regression_normal_results = pickle.load(f)
    
with open("preloaded_data/USO_2013_2025_results.pkl", "rb") as f:
    assymetric_regression_2013_2025_results = pickle.load(f)
    
with open("preloaded_data/USO_2013_2025rnd_results.pkl", "rb") as f:
        assymetric_regression_2013_2025rnd_results = pickle.load(f)
    
with open("preloaded_data/stock_df.pkl", "rb") as f:
    stock_df = pickle.load(f)

    


##################################################################Quantile regressions with bootstraps (will take a long time)####################################
for horizon in investment_horizon:

    assymetric_regression_2008_2019_results[horizon] = run_asym_quantreg_with_controls(

        # ----------------------------------------
        # DATA
        # ----------------------------------------
        r_df=logreturns,                       # dataframe containing returns
        var_s=moments_2008_2019_dict[horizon]["rnd_vol_ann"],
        skew_s=moments_2008_2019_dict[horizon]["rnd_skew"],
        kurt_s=moments_2008_2019_dict[horizon]["rnd_kurt"],

        # ----------------------------------------
        # RETURN COLUMN
        # ----------------------------------------
        ret_col="ret_1",

        # ----------------------------------------
        # MODEL LAGS
        # ----------------------------------------
        n_controls_lags=2,   # lags of moments as controls
        n_ret_lags=2,        # lags of positive/negative returns
        n_mom_lags=2,        # AR lags of dependent variable
        # ----------------------------------------
        # quantiles
        # ----------------------------------------
        taus=taus,
        segments =segments,

        

        # ----------------------------------------
        # BOOTSTRAP SETTINGS
        # ----------------------------------------
        B=10000,
        block_len=25,
        horizon_label=horizon,
        # ----------------------------------------
        # STORAGE OPTIONS
        # ----------------------------------------
        store_boot_params=True,
        store_wald_mats=True,
        equations_to_run= ["A_var","B_skew","C_kurt"],


        # ----------------------------------------
        # OPTIONAL PLOTTING OF TEST DISTRIBUTIONS
        # ----------------------------------------
        plot_test_dist=True,
        plot_test_dist_dir="results_tables_and_figures/gallery/bootstrap",
        plot_test_dist_prefix="2008"
    )
    
    assymetric_regression_crisis_results[horizon] = run_asym_quantreg_with_controls(
        r_df=logreturns,                       # dataframe containing returns
        var_s=moments_crisis_dict[horizon]["rnd_vol_ann"],
        skew_s=moments_crisis_dict[horizon]["rnd_skew"],
        kurt_s=moments_crisis_dict[horizon]["rnd_kurt"],
        ret_col="ret_1",
        n_controls_lags=2,   # lags of moments as controls
        n_ret_lags=2,        # lags of positive/negative returns
        n_mom_lags=2,        # AR lags of dependent variable
        taus=taus,
        segments =segments,
        B=10000,
        block_len=25,
        horizon_label=horizon,
        store_boot_params=True,
        store_wald_mats=True,
        equations_to_run= ["A_var","B_skew","C_kurt"],
        plot_test_dist=True,
        plot_test_dist_dir="results_tables_and_figures/gallery/bootstrap",
        plot_test_dist_prefix="crisis"
    )
    
    assymetric_regression_normal_results[horizon] = run_asym_quantreg_with_controls(
        r_df=logreturns,                       # dataframe containing returns
        var_s=moments_normal_dict[horizon]["rnd_vol_ann"],
        skew_s=moments_normal_dict[horizon]["rnd_skew"],
        kurt_s=moments_normal_dict[horizon]["rnd_kurt"],
        ret_col="ret_1",
        n_controls_lags=2,   # lags of moments as controls
        n_ret_lags=2,        # lags of positive/negative returns
        n_mom_lags=2,        # AR lags of dependent variable
        taus=taus,
        segments =segments,
        B=10000,
        block_len=25,
        horizon_label=horizon,
        store_boot_params=True,
        store_wald_mats=True,
        equations_to_run= ["A_var","B_skew","C_kurt"],
        plot_test_dist=True,
        plot_test_dist_dir="results_tables_and_figures/gallery/bootstrap",
        plot_test_dist_prefix="normal"
    )
    
    
    assymetric_regression_2013_2025_results[horizon]= run_asym_quantreg_with_controls(
        r_df=logreturns,            
        var_s=moments_2013_2025_dict[horizon]["phys_vol_ann"],
        skew_s=moments_2013_2025_dict[horizon]["phys_skew"],
        kurt_s=moments_2013_2025_dict[horizon]["phys_kurt"],
        ret_col="ret_1",
        horizon_label=horizon,
        n_controls_lags=2,
        n_ret_lags=2,
        n_mom_lags=2,
        B=10000,
        block_len=25,
        store_boot_params=True,
        store_wald_mats=True,
        plot_test_dist=True,
        plot_test_dist_dir="results_tables_and_figures/gallery/bootstrap",
        plot_test_dist_prefix="2013_phys"
)

    assymetric_regression_2013_2025rnd_results[horizon]= run_asym_quantreg_with_controls(
        r_df=logreturns,            
        var_s=moments_2013_2025_dict[horizon]["rnd_vol_ann"],
        skew_s=moments_2013_2025_dict[horizon]["rnd_skew"],
        kurt_s=moments_2013_2025_dict[horizon]["rnd_kurt"],
        ret_col="ret_1",
        horizon_label=horizon,
        n_controls_lags=2,
        n_ret_lags=2,
        n_mom_lags=2,
        B=10000,
        block_len=25,
        store_boot_params=True,
        store_wald_mats=True,
        plot_test_dist=True,
        plot_test_dist_dir="results_tables_and_figures/gallery/bootstrap",
        plot_test_dist_prefix="2013_rnd"
)


###############################################################Latex tables and Figures
qr_results = {
    "2008_2019": assymetric_regression_2008_2019_results,
    "crisis": assymetric_regression_crisis_results,
    "normal": assymetric_regression_normal_results,
    "2013_2025": assymetric_regression_2013_2025_results,
    "2013_2025rnd": assymetric_regression_2013_2025rnd_results
}

#Path("tables").mkdir(exist_ok=True)
#Path("latex_tests").mkdir(exist_ok=True)

table_specs = [
    {
        "eq_key": "A_var",
        "caption": "Quantile regression estimates for volatility across horizons.",
        "label": "tab:vol_qreg_allH",
        "save_stub": "table_quantile_regression_vol",
        "include_vars": [
            "ret_pos", "ret_neg",
            "ret_pos_L1", "ret_neg_L1",
            "ret_pos_L2", "ret_neg_L2",
            "d_var_L1", "d_var_L2",
        ],
        "var_rename": {
            "ret_pos": r"$Ret^{+}$",
            "ret_neg": r"$Ret^{-}$",
            "ret_pos_L1": r"$Ret^{+}_{t-1}$",
            "ret_neg_L1": r"$Ret^{-}_{t-1}$",
            "ret_pos_L2": r"$Ret^{+}_{t-2}$",
            "ret_neg_L2": r"$Ret^{-}_{t-2}$",
            "d_var_L1": r"$\Delta Vol_{t-1}$",
            "d_var_L2": r"$\Delta Vol_{t-2}$",
        },
        "include_const": False,
    },
    {
        "eq_key": "B_skew",
        "caption": "Quantile regression estimates for skewness across horizons.",
        "label": "tab:skew_qreg_allH",
        "save_stub": "table_quantile_regression_skew",
        "include_vars": [
            "ret_pos", "ret_neg",
            "ret_pos_L1", "ret_neg_L1",
            "ret_pos_L2", "ret_neg_L2",
            "d_skew_L1", "d_skew_L2",
        ],
        "var_rename": {
            "ret_pos": r"$Ret^{+}$",
            "ret_neg": r"$Ret^{-}$",
            "ret_pos_L1": r"$Ret^{+}_{t-1}$",
            "ret_neg_L1": r"$Ret^{-}_{t-1}$",
            "ret_pos_L2": r"$Ret^{+}_{t-2}$",
            "ret_neg_L2": r"$Ret^{-}_{t-2}$",
            "d_skew_L1": r"$\Delta Skew_{t-1}$",
            "d_skew_L2": r"$\Delta Skew_{t-2}$",
        },
        "include_const": False,
    },
    {
        "eq_key": "C_kurt",
        "caption": "Quantile regression estimates for kurtosis across horizons.",
        "label": "tab:kurt_qreg_allH",
        "save_stub": "table_quantile_regression_kurt",
        "include_vars": [
            "ret_pos", "ret_neg",
            "ret_pos_L1", "ret_neg_L1",
            "ret_pos_L2", "ret_neg_L2",
            "d_kurt_L1", "d_kurt_L2",
        ],
        "var_rename": {
            "ret_pos": r"$Ret^{+}$",
            "ret_neg": r"$Ret^{-}$",
            "ret_pos_L1": r"$Ret^{+}_{t-1}$",
            "ret_neg_L1": r"$Ret^{-}_{t-1}$",
            "ret_pos_L2": r"$Ret^{+}_{t-2}$",
            "ret_neg_L2": r"$Ret^{-}_{t-2}$",
            "d_kurt_L1": r"$\Delta Kurt_{t-1}$",
            "d_kurt_L2": r"$\Delta Kurt_{t-2}$",
        },
        "include_const": False,
    },
]


base_dir = Path.cwd()



qr_tables_dir = base_dir / "results_tables_and_figures"
hypothesis_dir = base_dir / "results_tables_and_figures"



qr_tables_dir.mkdir(parents=True, exist_ok=True)
hypothesis_dir.mkdir(parents=True, exist_ok=True)

all_tex={}
for sample_name, model in qr_results.items():
    all_tex[sample_name] = {}

    # -----------------------------
    # coefficient tables
    # -----------------------------
    for spec in table_specs:
        tex = asym_to_latex_table(
                    model,
                    eq_key=spec["eq_key"],
                    caption=spec["caption"],
                    label=f'{spec["label"]}_{sample_name}',
                    include_vars=spec["include_vars"],
                    var_rename=spec["var_rename"],
                    include_ols=True,
                    include_const=spec.get("include_const", False),
                    show_pvalues_below=False,
                    decimals=4,
                    taus=[0.05,0.1,0.25,0.5,0.75,0.9,0.95],
                    size_cmd=r"\tiny",
                    save_path=str(qr_tables_dir / f'{spec["save_stub"]}_{sample_name}.tex'),
                )
        all_tex[sample_name][spec["eq_key"]] = tex

    # -----------------------------
    # hypothesis test tables
    # -----------------------------
    test_tables = build_asym_quantreg_tests_latex(
        model,
        base_dir=hypothesis_dir / f"Hypothesis_table_{sample_name}",
        file_prefix=f"test_table_{sample_name}",
    )

    all_tex[sample_name]["hypothesis_tests"] = test_tables

#################PLots
res = {
    "7d": assymetric_regression_2008_2019_results[7],
    "14d": assymetric_regression_2008_2019_results[14],
    "21d": assymetric_regression_2008_2019_results[21],
    "28d": assymetric_regression_2008_2019_results[28],
    "35d": assymetric_regression_2008_2019_results[35],
    "60d": assymetric_regression_2008_2019_results[60],

}

title_map = {
    "A_var": "Annualized Risk Neutral Volatility",
    "B_skew": "Risk Neutral Skewness",
    "C_kurt": "Risk Neutral Kurtosis",
}

key_list = ["A_var", "B_skew", "C_kurt"]

for key in key_list:
    base_title = title_map[key]

    fig, axes = plot_qrm_across_quantiles_selectcoef(
        res_by_key=res,
        eq_key=key,
        coefs=("ret_pos", "ret_neg"),
        ci=0.95,
        show_ols=True,
        title=f"{base_title} Coefficients Across Quantiles and Horizons",
        save_path=f"results_tables_and_figures/figures/{key}_quantiles.png",
        dpi=300,
        close_after_save=True,
    )
        
    fig, axes = plot_qrm_by_quantile_across_frequencies_selectcoef(
        res_by_key=res,
        eq_key=key,
        coefs=("ret_pos", "ret_neg"),
        taus_to_plot=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        ci=0.95,
        title=f"{base_title} by Quantile Across Horizons",
        save_path=f"results_tables_and_figures/figures/{key}_by_tau_across_horizons.png",
        dpi=300,
        close_after_save=True,
    )


