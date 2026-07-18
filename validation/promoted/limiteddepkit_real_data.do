version 17.0
clear all
set more off
set linesize 255

/*
Promoted real-data parity runner for Stata 17 or newer.

Prepare the work directory with prepare_real_data.py, and then run, for example,

    do "validation/promoted/limiteddepkit_real_data.do" ///
        "C:/path/to/limiteddepkit/validation/promoted/work/real_data"

The script never installs community commands.  `firthlogit` is used only when
it is already installed.  Gamma duration is an explicit Stata skip because
`streg`'s generalized-gamma model is not the ordinary censored Gamma likelihood
implemented by limiteddepkit.
*/

args workdir
if `"`workdir'"' == "" {
    display as error "Pass the promoted real-data work directory as argument 1."
    exit 198
}

local data_dir `"`workdir'/data"'
local output_dir `"`workdir'/stata"'
capture mkdir `"`output_dir'"'

foreach fixture in binary_lbw count_rod93 censoring_mroz87 ///
        censoring_womenwage2 duration_cancer duration_cancer_deaths ///
        ordinal_tvsfpors panel_nlswork {
    capture confirm file `"`data_dir'/`fixture'.dta"'
    if _rc {
        display as error "Missing prepared fixture: `data_dir'/`fixture'.dta"
        exit 601
    }
}

capture log close
foreach artifact in stata_run.log estimates_raw.csv covariance_raw.csv fit.csv ///
        predictions.csv model_status.csv metadata.txt estimates_canonical.csv ///
        covariance_canonical.csv {
    capture erase `"`output_dir'/`artifact'"'
}
foreach artifact in comparison_report.csv comparison_summary.md ///
        parity_certificate.json {
    capture erase `"`workdir'/`artifact'"'
}
log using `"`output_dir'/stata_run.log"', text replace

tempname coefficient_handle covariance_handle fit_handle prediction_handle ///
    status_handle metadata_handle
tempfile coefficient_data covariance_data fit_data prediction_data status_data

postfile `coefficient_handle' str48 model str48 dataset int position ///
    str244 stata_parameter double estimate double standard_error ///
    using `"`coefficient_data'"', replace
postfile `covariance_handle' str48 model str48 dataset int row_position ///
    int column_position str244 row_parameter str244 column_parameter ///
    double covariance using `"`covariance_data'"', replace
postfile `fit_handle' str48 model str48 dataset double nobs n_params ///
    loglike aic bic converged inference_valid n_groups n_events n_censored ///
    n_interval n_exact n_left_censored n_right_censored score_norm ///
    scaled_score_norm penalized_loglike jeffreys_penalty n_cutoff_clones ///
    n_pseudo_observations n_entities n_contributing_entities ///
    str64 backend str64 covariance_type ///
    using `"`fit_data'"', replace
postfile `prediction_handle' str48 model str48 dataset long obs_id ///
    str32 prediction double category time value ///
    using `"`prediction_data'"', replace
postfile `status_handle' str48 model str48 dataset str16 status ///
    str160 reason using `"`status_data'"', replace

capture program drop ldk_post_estimation
program define ldk_post_estimation
    version 17.0
    args coefficient_handle covariance_handle model_name dataset_name

    tempname coefficients variance
    matrix `coefficients' = e(b)
    matrix `variance' = e(V)
    local parameter_names : colfullnames `coefficients'
    local n_parameters = colsof(`coefficients')

    forvalues column = 1/`n_parameters' {
        local parameter : word `column' of `parameter_names'
        scalar __ldk_estimate = el(`coefficients', 1, `column')
        scalar __ldk_variance = el(`variance', `column', `column')
        scalar __ldk_standard_error = ///
            cond(__ldk_variance >= 0, sqrt(__ldk_variance), .)
        post `coefficient_handle' ("`model_name'") ("`dataset_name'") ///
            (`column') ("`parameter'") (__ldk_estimate) ///
            (__ldk_standard_error)
    }

    forvalues row = 1/`n_parameters' {
        local row_parameter : word `row' of `parameter_names'
        forvalues column = 1/`n_parameters' {
            local column_parameter : word `column' of `parameter_names'
            scalar __ldk_covariance = el(`variance', `row', `column')
            post `covariance_handle' ("`model_name'") ("`dataset_name'") ///
                (`row') (`column') ("`row_parameter'") ///
                ("`column_parameter'") (__ldk_covariance)
        }
    }
end

capture program drop ldk_post_fit
program define ldk_post_fit
    version 17.0
    syntax anything(name=fit_handle), MODEL(string) DATASET(string) ///
        [NOBS(real -1) NGROUPS(real -1) NEVENTS(real -1) ///
        NCENSORED(real -1) NINTERVAL(real -1) NEXACT(real -1) ///
        NLEFT(real -1) NRIGHT(real -1) NCLONES(real -1) ///
        NPSEUDO(real -1) NENTITIES(real -1) NCONTRIB(real -1) ///
        BACKEND(string) COVARIANCE(string) NOIC]

    scalar __ldk_nobs = e(N)
    if `nobs' >= 0 scalar __ldk_nobs = `nobs'
    scalar __ldk_nparams = colsof(e(b))
    capture scalar __ldk_nparams = e(rank)
    scalar __ldk_loglike = e(ll)
    scalar __ldk_converged = 1
    capture scalar __ldk_converged = e(converged)
    scalar __ldk_aic = .
    scalar __ldk_bic = .
    if `"`noic'"' == "" {
        scalar __ldk_aic = -2 * __ldk_loglike + 2 * __ldk_nparams
        scalar __ldk_bic = ///
            -2 * __ldk_loglike + ln(__ldk_nobs) * __ldk_nparams
    }

    scalar __ldk_ngroups = .
    if `ngroups' >= 0 scalar __ldk_ngroups = `ngroups'
    scalar __ldk_nevents = cond(`nevents' >= 0, `nevents', .)
    scalar __ldk_ncensored = cond(`ncensored' >= 0, `ncensored', .)
    scalar __ldk_ninterval = cond(`ninterval' >= 0, `ninterval', .)
    scalar __ldk_nexact = cond(`nexact' >= 0, `nexact', .)
    scalar __ldk_nleft = cond(`nleft' >= 0, `nleft', .)
    scalar __ldk_nright = cond(`nright' >= 0, `nright', .)
    scalar __ldk_nclones = cond(`nclones' >= 0, `nclones', .)
    scalar __ldk_npseudo = cond(`npseudo' >= 0, `npseudo', .)
    scalar __ldk_nentities = cond(`nentities' >= 0, `nentities', .)
    scalar __ldk_ncontrib = cond(`ncontrib' >= 0, `ncontrib', .)

    local backend_value `"`backend'"'
    if `"`backend_value'"' == "" local backend_value `"`e(cmd)'"'
    local covariance_value `"`covariance'"'
    if `"`covariance_value'"' == "" local covariance_value `"`e(vce)'"'

    post `fit_handle' ("`model'") ("`dataset'") (__ldk_nobs) ///
        (__ldk_nparams) (__ldk_loglike) (__ldk_aic) (__ldk_bic) ///
        (__ldk_converged) (1) (__ldk_ngroups) (__ldk_nevents) ///
        (__ldk_ncensored) (__ldk_ninterval) (__ldk_nexact) ///
        (__ldk_nleft) (__ldk_nright) (.) (.) (.) (.) (__ldk_nclones) ///
        (__ldk_npseudo) (__ldk_nentities) (__ldk_ncontrib) ///
        ("`backend_value'") ("`covariance_value'")
end

capture program drop ldk_post_predictions
program define ldk_post_predictions
    version 17.0
    syntax anything(name=prediction_handle), MODEL(string) DATASET(string) ///
        ID(varname numeric) PREDICTION(string) VALUE(varname numeric) ///
        [CATEGORY(varname numeric) TIME(varname numeric)]

    forvalues row = 1/`=_N' {
        scalar __ldk_category = .
        scalar __ldk_time = .
        if `"`category'"' != "" scalar __ldk_category = `category'[`row']
        if `"`time'"' != "" scalar __ldk_time = `time'[`row']
        post `prediction_handle' ("`model'") ("`dataset'") ///
            (`id'[`row']) ("`prediction'") (__ldk_category) ///
            (__ldk_time) (`value'[`row'])
    }
end

/* Optional Firth Binary Logit. */
capture which firthlogit
local have_firthlogit = (_rc == 0)
local firthlogit_path "not_installed"
if `have_firthlogit' {
    capture quietly findfile firthlogit.ado
    if !_rc local firthlogit_path `"`r(fn)'"'

    display as text "Running optional firthlogit parity fit"
    use `"`data_dir'/binary_lbw.dta"', clear
    capture noisily firthlogit y x1 x2 x3 x4
    if _rc {
        local fit_rc = _rc
        display as error "Installed firthlogit failed; this is not an allowed skip."
        log close
        exit `fit_rc'
    }
    ldk_post_estimation `coefficient_handle' `covariance_handle' ///
        firth_binary_logit binary_lbw
    ldk_post_fit `fit_handle', model(firth_binary_logit) ///
        dataset(binary_lbw) backend(firthlogit) covariance(native) noic

    predict double __firth_xb, xb
    generate double __firth_p1 = invlogit(__firth_xb)
    generate double __firth_p0 = 1 - __firth_p1
    generate byte __firth_category = 0
    ldk_post_predictions `prediction_handle', model(firth_binary_logit) ///
        dataset(binary_lbw) id(obs_id) prediction(probability) ///
        value(__firth_p0) category(__firth_category)
    replace __firth_category = 1
    ldk_post_predictions `prediction_handle', model(firth_binary_logit) ///
        dataset(binary_lbw) id(obs_id) prediction(probability) ///
        value(__firth_p1) category(__firth_category)
    post `status_handle' ("firth_binary_logit") ("binary_lbw") ///
        ("RUN") ("optional firthlogit command available")
}
else {
    display as result "firthlogit is not installed; optional Stata fit skipped."
    display as result "Install manually if desired: ssc install firthlogit"
    post `status_handle' ("firth_binary_logit") ("binary_lbw") ///
        ("SKIP") ("optional_command_not_installed")
}

/* Poisson and NB2 count models, including exposure. */
display as text "Running Poisson and NB2 parity fits"
use `"`data_dir'/count_rod93.dta"', clear
quietly poisson deaths intercept log_age cohort_2 cohort_3, ///
    noconstant exposure(exposure) vce(oim) iterate(3000) tolerance(1e-10)
ldk_post_estimation `coefficient_handle' `covariance_handle' poisson count_rod93
ldk_post_fit `fit_handle', model(poisson) dataset(count_rod93) ///
    backend(poisson) covariance(oim)
predict double __poisson_mean, n
ldk_post_predictions `prediction_handle', model(poisson) ///
    dataset(count_rod93) id(obs_id) prediction(mean) value(__poisson_mean)
post `status_handle' ("poisson") ("count_rod93") ("RUN") ///
    ("official poisson with exposure")

quietly nbreg deaths intercept log_age cohort_2 cohort_3, ///
    noconstant exposure(exposure) dispersion(mean) vce(oim) ///
    iterate(3000) tolerance(1e-10)
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    negative_binomial_nb2 count_rod93
ldk_post_fit `fit_handle', model(negative_binomial_nb2) ///
    dataset(count_rod93) backend(nbreg_nb2) covariance(oim)
predict double __nb2_mean, n
ldk_post_predictions `prediction_handle', model(negative_binomial_nb2) ///
    dataset(count_rod93) id(obs_id) prediction(mean) value(__nb2_mean)
post `status_handle' ("negative_binomial_nb2") ("count_rod93") ///
    ("RUN") ("official nbreg dispersion(mean) with exposure")

/* Left-censored Tobit and positive-sample left-truncated regression. */
display as text "Running Tobit and truncated-regression parity fits"
use `"`data_dir'/censoring_mroz87.dta"', clear
quietly count if y <= 0
local tobit_censored = r(N)
quietly count if y > 0
local tobit_exact = r(N)
quietly tobit y intercept age_z education_z experience_z young_children, ///
    noconstant ll(0) vce(oim) iterate(3000) tolerance(1e-10)
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    tobit censoring_mroz87
ldk_post_fit `fit_handle', model(tobit) dataset(censoring_mroz87) ///
    ncensored(`tobit_censored') nexact(`tobit_exact') ///
    backend(tobit) covariance(oim)
predict double __tobit_mean, ystar(0,.)
ldk_post_predictions `prediction_handle', model(tobit) ///
    dataset(censoring_mroz87) id(obs_id) prediction(mean) value(__tobit_mean)
post `status_handle' ("tobit") ("censoring_mroz87") ("RUN") ///
    ("official tobit ll(0)")

use `"`data_dir'/censoring_mroz87.dta"', clear
keep if positive == 1
quietly truncreg y intercept age_z education_z experience_z young_children, ///
    noconstant ll(0) vce(oim) iterate(3000) tolerance(1e-10)
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    truncated_regression censoring_mroz87
ldk_post_fit `fit_handle', model(truncated_regression) ///
    dataset(censoring_mroz87) backend(truncreg) covariance(oim)
predict double __truncated_mean, e(0,.)
ldk_post_predictions `prediction_handle', model(truncated_regression) ///
    dataset(censoring_mroz87) id(obs_id) prediction(mean) ///
    value(__truncated_mean)
post `status_handle' ("truncated_regression") ("censoring_mroz87") ///
    ("RUN") ("official truncreg ll(0), positive subsample")

/* Open-ended interval regression. */
display as text "Running interval-regression parity fit"
use `"`data_dir'/censoring_womenwage2.dta"', clear
quietly count if !missing(lower) & !missing(upper) & lower < upper
local interval_n = r(N)
quietly count if !missing(lower) & !missing(upper) & lower == upper
local exact_n = r(N)
quietly count if missing(lower) & !missing(upper)
local left_n = r(N)
quietly count if !missing(lower) & missing(upper)
local right_n = r(N)
quietly intreg lower upper intercept age_z school_z tenure_z ///
    never_married rural, noconstant vce(oim) iterate(3000) tolerance(1e-10)
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    interval_regression censoring_womenwage2
ldk_post_fit `fit_handle', model(interval_regression) ///
    dataset(censoring_womenwage2) ninterval(`interval_n') ///
    nexact(`exact_n') nleft(`left_n') nright(`right_n') ///
    backend(intreg) covariance(oim)
predict double __interval_mean, xb
ldk_post_predictions `prediction_handle', model(interval_regression) ///
    dataset(censoring_womenwage2) id(obs_id) prediction(mean) ///
    value(__interval_mean)
post `status_handle' ("interval_regression") ("censoring_womenwage2") ///
    ("RUN") ("official intreg with missing open endpoints")

/* Geometric duration: exact person-period likelihood identity. */
display as text "Running geometric-duration person-period parity fit"
use `"`data_dir'/duration_cancer.dta"', clear
isid obs_id
assert duration >= 1 & duration == floor(duration)
quietly count
local geometric_nobs = r(N)
quietly summarize event, meanonly
local geometric_events = r(sum)
expand duration
bysort obs_id: generate long __period = _n
generate byte __event_pp = event == 1 & __period == duration
quietly logit __event_pp intercept age_z drug_2 drug_3, ///
    noconstant vce(oim) iterate(3000) tolerance(1e-10)
local geometric_pseudo = e(N)
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    geometric_duration duration_cancer
ldk_post_fit `fit_handle', model(geometric_duration) ///
    dataset(duration_cancer) nobs(`geometric_nobs') ///
    nevents(`geometric_events') npseudo(`geometric_pseudo') ///
    backend(person_period_logit_identity) covariance(oim)
predict double __geometric_hazard, pr
generate double __geometric_mean = 1 / __geometric_hazard
generate double __geometric_s5 = (1 - __geometric_hazard)^5
generate double __geometric_s15 = (1 - __geometric_hazard)^15
generate double __prediction_time = 5
preserve
keep if __period == 1
ldk_post_predictions `prediction_handle', model(geometric_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(mean) ///
    value(__geometric_mean)
ldk_post_predictions `prediction_handle', model(geometric_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(survival) ///
    value(__geometric_s5) time(__prediction_time)
replace __prediction_time = 15
ldk_post_predictions `prediction_handle', model(geometric_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(survival) ///
    value(__geometric_s15) time(__prediction_time)
restore
post `status_handle' ("geometric_duration") ("duration_cancer") ///
    ("RUN") ("exact person-period logit likelihood identity")

/* Exponential and Weibull AFT fits. */
display as text "Running continuous-duration AFT parity fits"
use `"`data_dir'/duration_cancer.dta"', clear
quietly count
local duration_nobs = r(N)
quietly summarize event, meanonly
local duration_events = r(sum)
stset duration, failure(event)
quietly streg intercept age_z drug_2 drug_3, distribution(exponential) ///
    time noconstant vce(oim) iterate(3000) tolerance(1e-10)
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    exponential_duration duration_cancer
ldk_post_fit `fit_handle', model(exponential_duration) ///
    dataset(duration_cancer) nobs(`duration_nobs') ///
    nevents(`duration_events') backend(streg_exponential_time) covariance(oim)
predict double __exponential_mean, mean time
generate double __saved_t = _t
replace _t = 5
predict double __exponential_s5, surv
replace _t = 15
predict double __exponential_s15, surv
replace _t = __saved_t
generate double __duration_prediction_time = 5
ldk_post_predictions `prediction_handle', model(exponential_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(mean) ///
    value(__exponential_mean)
ldk_post_predictions `prediction_handle', model(exponential_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(survival) ///
    value(__exponential_s5) time(__duration_prediction_time)
replace __duration_prediction_time = 15
ldk_post_predictions `prediction_handle', model(exponential_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(survival) ///
    value(__exponential_s15) time(__duration_prediction_time)
post `status_handle' ("exponential_duration") ("duration_cancer") ///
    ("RUN") ("official streg exponential time metric")

use `"`data_dir'/duration_cancer.dta"', clear
stset duration, failure(event)
quietly streg intercept age_z drug_2 drug_3, distribution(weibull) ///
    time noconstant vce(oim) iterate(3000) tolerance(1e-10)
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    weibull_duration duration_cancer
ldk_post_fit `fit_handle', model(weibull_duration) ///
    dataset(duration_cancer) nobs(`duration_nobs') ///
    nevents(`duration_events') backend(streg_weibull_time) covariance(oim)
predict double __weibull_mean, mean time
generate double __weibull_saved_t = _t
replace _t = 5
predict double __weibull_s5, surv
replace _t = 15
predict double __weibull_s15, surv
replace _t = __weibull_saved_t
generate double __weibull_prediction_time = 5
ldk_post_predictions `prediction_handle', model(weibull_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(mean) value(__weibull_mean)
ldk_post_predictions `prediction_handle', model(weibull_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(survival) ///
    value(__weibull_s5) time(__weibull_prediction_time)
replace __weibull_prediction_time = 15
ldk_post_predictions `prediction_handle', model(weibull_duration) ///
    dataset(duration_cancer) id(obs_id) prediction(survival) ///
    value(__weibull_s15) time(__weibull_prediction_time)
post `status_handle' ("weibull_duration") ("duration_cancer") ///
    ("RUN") ("official streg Weibull time metric; /ln_p retained raw")

post `status_handle' ("gamma_duration") ("duration_cancer_deaths") ///
    ("SKIP") ///
    ("unsupported_exact_match: streg ggamma is not ordinary censored Gamma")

/* Random-intercept Ordered Probit, fixed random effect prediction b=0. */
display as text "Running random-effects Ordered-Probit parity fit"
use `"`data_dir'/ordinal_tvsfpors.dta"', clear
egen byte __entity_tag = tag(entity)
quietly count if __entity_tag
local re_groups = r(N)
drop __entity_tag
matrix __re_start = ( ///
    0.23698070900211501, ///
    0.54905174772184595, ///
    0.16948874153822913, ///
    -0.29513785838283513, ///
    -0.068250830319126043, ///
    0.67675982721408678, ///
    1.3905987368670569, ///
    0.028849194266371211)
matrix colnames __re_start = y:x1 y:x2 y:x3 y:x4 ///
    /cut1 /cut2 /cut3 /var(_cons[entity])
quietly meoprobit y x1 x2 x3 x4 || entity:, ///
    intmethod(ghermite) intpoints(20) vce(oim) ///
    iterate(3000) tolerance(1e-10) nrtolerance(1e-8) ///
    from(__re_start) difficult
local re_intmethod `"`e(intmethod)'"'
local re_n_quad `"`e(n_quad)'"'
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    random_effects_ordered_probit ordinal_tvsfpors
ldk_post_fit `fit_handle', model(random_effects_ordered_probit) ///
    dataset(ordinal_tvsfpors) ngroups(`re_groups') ///
    backend(meoprobit_ghermite_q20) covariance(oim) noic
predict double __re_p0 __re_p1 __re_p2 __re_p3, ///
    pr conditional(fixedonly)
generate byte __re_category = 0
foreach category in 0 1 2 3 {
    ldk_post_predictions `prediction_handle', ///
        model(random_effects_ordered_probit) dataset(ordinal_tvsfpors) ///
        id(obs_id) prediction(probability) value(__re_p`category') ///
        category(__re_category)
    replace __re_category = __re_category + 1
}
post `status_handle' ("random_effects_ordered_probit") ///
    ("ordinal_tvsfpors") ("RUN") ///
    ("official meoprobit nonadaptive Gauss-Hermite Q=20")

/* BUC Ordered Logit: exact blow-up, conditional logit, entity clustering. */
display as text "Running BUC fixed-effects Ordered-Logit parity fit"
use `"`data_dir'/panel_nlswork.dta"', clear
isid obs_id
quietly count
local fe_nobs = r(N)
egen byte __all_entity_tag = tag(entity)
quietly count if __all_entity_tag
local fe_entities = r(N)
drop __all_entity_tag

expand 2
bysort obs_id: generate byte __cutoff = _n
generate byte __buc_outcome = y >= __cutoff
egen long __entity_cutoff = group(entity __cutoff)
bysort __entity_cutoff: egen byte __clone_min = min(__buc_outcome)
bysort __entity_cutoff: egen byte __clone_max = max(__buc_outcome)
keep if __clone_min < __clone_max
egen byte __clone_tag = tag(__entity_cutoff)
quietly count if __clone_tag
local fe_clones = r(N)
egen byte __contributing_entity_tag = tag(entity)
quietly count if __contributing_entity_tag
local fe_contributing = r(N)
quietly count
local fe_pseudo = r(N)

quietly clogit __buc_outcome x1, group(__entity_cutoff) ///
    vce(cluster entity) iterate(3000) tolerance(1e-10)
ldk_post_estimation `coefficient_handle' `covariance_handle' ///
    fixed_effects_ordered_logit panel_nlswork
ldk_post_fit `fit_handle', model(fixed_effects_ordered_logit) ///
    dataset(panel_nlswork) nobs(`fe_nobs') ngroups(`fe_entities') ///
    nclones(`fe_clones') npseudo(`fe_pseudo') nentities(`fe_entities') ///
    ncontrib(`fe_contributing') backend(clogit_buc) ///
    covariance(entity_cluster_sandwich) noic
post `status_handle' ("fixed_effects_ordered_logit") ///
    ("panel_nlswork") ("RUN") ///
    ("exact BUC blow-up/clogit; covariance clustered by original entity")

postclose `coefficient_handle'
postclose `covariance_handle'
postclose `fit_handle'
postclose `prediction_handle'
postclose `status_handle'

use `"`coefficient_data'"', clear
sort model position
format estimate standard_error %24.17g
export delimited using `"`output_dir'/estimates_raw.csv"', ///
    replace datafmt

use `"`covariance_data'"', clear
sort model row_position column_position
format covariance %24.17g
export delimited using `"`output_dir'/covariance_raw.csv"', ///
    replace datafmt

use `"`fit_data'"', clear
sort model
format nobs-n_contributing_entities %24.17g
export delimited using `"`output_dir'/fit.csv"', replace datafmt

use `"`prediction_data'"', clear
sort model obs_id prediction category time
format category time value %24.17g
export delimited using `"`output_dir'/predictions.csv"', ///
    replace datafmt

use `"`status_data'"', clear
sort model
export delimited using `"`output_dir'/model_status.csv"', ///
    replace datafmt

file open `metadata_handle' using `"`output_dir'/metadata.txt"', ///
    write text replace
file write `metadata_handle' "suite=promoted_public_data_parity" _n
file write `metadata_handle' "stata_version=`c(stata_version)'" _n
file write `metadata_handle' "firthlogit_installed=`have_firthlogit'" _n
file write `metadata_handle' `"firthlogit_path=`firthlogit_path'"' _n
file write `metadata_handle' `"random_effects_ordered_probit.intmethod=`re_intmethod'"' _n
file write `metadata_handle' `"random_effects_ordered_probit.n_quad=`re_n_quad'"' _n
file write `metadata_handle' "random_effects_prediction=conditional_fixedonly_b0" _n
file write `metadata_handle' "geometric_evidence=pseudo_sample_likelihood_identity" _n
file write `metadata_handle' "fixed_effects_evidence=conditional_buc_pseudo_sample" _n
file write `metadata_handle' "gamma_duration=unsupported_exact_stata_match" _n
file write `metadata_handle' "interval_fixture=official_fictional_software_example" _n
file write `metadata_handle' "run_completed=1" _n
file close `metadata_handle'

display as result "Promoted Stata outputs written to `output_dir'"
display as result "Run compare_stata.py on `workdir' to evaluate parity."
log close
