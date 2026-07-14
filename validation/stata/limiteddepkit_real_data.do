version 15.1
clear all
set more off
set linesize 255

/*
Run from Stata with the real-data work directory produced by the Python
preparation script:

    do "validation/stata/limiteddepkit_real_data.do" ///
        "C:/path/to/limiteddepkit/validation/stata/work/real_data"

The script uses official Stata commands except for the two optional
gologit2 fits. It never installs community commands automatically.
*/

args workdir
if `"`workdir'"' == "" {
    display as error "Pass the prepared real-data work directory as the first argument."
    exit 198
}

local data_dir `"`workdir'/data"'
local output_dir `"`workdir'/stata"'
capture mkdir `"`output_dir'"'

foreach fixture in binary_lbw ordinal_tvsfpors dynamic_nlswork_design {
    capture confirm file `"`data_dir'/`fixture'.dta"'
    if _rc {
        display as error "Missing `data_dir'/`fixture'.dta"
        display as error "Run the real-data preparation script before this do-file."
        exit 601
    }
}

capture log close
foreach artifact in stata_run.log estimates_raw.csv covariance_raw.csv fit.csv ///
        predictions.csv metadata.txt estimates_canonical.csv covariance_canonical.csv {
    capture erase `"`output_dir'/`artifact'"'
}
foreach artifact in comparison_report.csv comparison_summary.md parity_certificate.json {
    capture erase `"`workdir'/`artifact'"'
}
log using `"`output_dir'/stata_run.log"', text replace

tempname coefficient_handle covariance_handle fit_handle prediction_handle metadata_handle
tempfile coefficient_data covariance_data fit_data prediction_data

postfile `coefficient_handle' str40 model int position str244 stata_parameter ///
    double estimate double standard_error using `"`coefficient_data'"', replace
postfile `covariance_handle' str40 model int row_position int column_position ///
    str244 row_parameter str244 column_parameter double covariance ///
    using `"`covariance_data'"', replace
postfile `fit_handle' str40 model double nobs double n_groups double n_params ///
    double loglike double aic double bic double converged ///
    using `"`fit_data'"', replace
postfile `prediction_handle' str40 model long obs_id int category double probability ///
    using `"`prediction_data'"', replace

capture program drop ldk_post_current
program define ldk_post_current
    version 15.1
    args coefficient_handle covariance_handle fit_handle model_name

    tempname coefficients variance group_counts
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
        post `coefficient_handle' ("`model_name'") (`column') ///
            ("`parameter'") (__ldk_estimate) (__ldk_standard_error)
    }

    forvalues row = 1/`n_parameters' {
        local row_parameter : word `row' of `parameter_names'
        forvalues column = 1/`n_parameters' {
            local column_parameter : word `column' of `parameter_names'
            scalar __ldk_covariance = el(`variance', `row', `column')
            post `covariance_handle' ("`model_name'") (`row') (`column') ///
                ("`row_parameter'") ("`column_parameter'") (__ldk_covariance)
        }
    }

    scalar __ldk_n_parameters = `n_parameters'
    capture scalar __ldk_n_parameters = e(rank)
    scalar __ldk_n_groups = .
    capture scalar __ldk_n_groups = e(N_g)
    capture matrix `group_counts' = e(N_g)
    if !_rc {
        scalar __ldk_n_groups = el(`group_counts', 1, 1)
    }
    scalar __ldk_converged = .
    capture scalar __ldk_converged = e(converged)
    scalar __ldk_aic = -2 * e(ll) + 2 * __ldk_n_parameters
    scalar __ldk_bic = -2 * e(ll) + ln(e(N)) * __ldk_n_parameters
    post `fit_handle' ("`model_name'") (e(N)) (__ldk_n_groups) ///
        (__ldk_n_parameters) (e(ll)) (__ldk_aic) (__ldk_bic) ///
        (__ldk_converged)
end

capture program drop ldk_post_predictions
program define ldk_post_predictions
    version 15.1
    syntax anything(name=prediction_handle), Model(string) ID(varname numeric) ///
        Probabilities(varlist numeric)

    local category = 0
    foreach probability of varlist `probabilities' {
        forvalues row = 1/`=_N' {
            scalar __ldk_obs_id = `id'[`row']
            scalar __ldk_probability = `probability'[`row']
            post `prediction_handle' ("`model'") (__ldk_obs_id) (`category') ///
                (__ldk_probability)
        }
        local category = `category' + 1
    }
end

display as text "Running LBW binary application fits"
use `"`data_dir'/binary_lbw.dta"', clear

quietly logit y intercept x1 x2 x3 x4, noconstant vce(oim)
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' binary_logit
predict double p1, pr
generate double p0 = 1 - p1
ldk_post_predictions `prediction_handle', model(binary_logit) id(obs_id) ///
    probabilities(p0 p1)
drop p0 p1

quietly probit y intercept x1 x2 x3 x4, noconstant vce(oim)
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' binary_probit
predict double p1, pr
generate double p0 = 1 - p1
ldk_post_predictions `prediction_handle', model(binary_probit) id(obs_id) ///
    probabilities(p0 p1)
drop p0 p1

display as text "Running TVSFPORS pooled ordinal application fits"
use `"`data_dir'/ordinal_tvsfpors.dta"', clear

quietly ologit y ox1 ox2 ox3 ox4, vce(oim)
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ordered_logit
predict double p0, outcome(#1)
predict double p1, outcome(#2)
predict double p2, outcome(#3)
predict double p3, outcome(#4)
ldk_post_predictions `prediction_handle', model(ordered_logit) id(obs_id) ///
    probabilities(p0 p1 p2 p3)
drop p0 p1 p2 p3

quietly oprobit y ox1 ox2 ox3 ox4, vce(oim)
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ordered_probit
predict double p0, outcome(#1)
predict double p1, outcome(#2)
predict double p2, outcome(#3)
predict double p3, outcome(#4)
ldk_post_predictions `prediction_handle', model(ordered_probit) id(obs_id) ///
    probabilities(p0 p1 p2 p3)
drop p0 p1 p2 p3

capture which gologit2
local have_gologit2 = (_rc == 0)
local gologit2_path "not_installed"
if `have_gologit2' {
    which gologit2
    capture quietly findfile gologit2.ado
    if !_rc {
        local gologit2_path `"`r(fn)'"'
    }

    display as text "Running optional gologit2 application fits"

    quietly gologit2 y gx1 gx2 gx3 gx4, npl
    ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ///
        generalized_ordered_logit
    predict double p0 p1 p2 p3
    ldk_post_predictions `prediction_handle', model(generalized_ordered_logit) ///
        id(obs_id) probabilities(p0 p1 p2 p3)
    drop p0 p1 p2 p3

    quietly gologit2 y gx1 gx2 gx3 gx4, npl(gx4)
    ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ///
        partial_proportional_odds
    predict double p0 p1 p2 p3
    ldk_post_predictions `prediction_handle', model(partial_proportional_odds) ///
        id(obs_id) probabilities(p0 p1 p2 p3)
    drop p0 p1 p2 p3
}
else {
    display as result "gologit2 is not installed; flexible ordinal fits were skipped."
    display as result "Install it manually with: ssc install gologit2"
}

display as text "Running TVSFPORS random-effects Ordered Logit application fit"
use `"`data_dir'/ordinal_tvsfpors.dta"', clear
quietly meologit y x1 x2 x3 x4 || entity:, ///
    intmethod(ghermite) intpoints(20) vce(oim) ///
    iterate(2000) tolerance(1e-10) ltolerance(1e-12) nrtolerance(1e-8)
local static_re_intmethod `"`e(intmethod)'"'
local static_re_n_quad `"`e(n_quad)'"'
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ///
    random_effects_ordered_logit
predict double p0 p1 p2 p3, pr conditional(fixedonly)
ldk_post_predictions `prediction_handle', model(random_effects_ordered_logit) ///
    id(obs_id) probabilities(p0 p1 p2 p3)
drop p0 p1 p2 p3

display as text "Running NLSWORK dynamic augmented-design application fit"
use `"`data_dir'/dynamic_nlswork_design.dta"', clear
quietly meologit y x1 state_1 state_2 initial_1 initial_2 initial_x1 mean_x1 ///
    || entity:, intmethod(ghermite) intpoints(20) vce(oim) ///
    iterate(2000) tolerance(1e-10) ltolerance(1e-12) nrtolerance(1e-8)
local dynamic_re_intmethod `"`e(intmethod)'"'
local dynamic_re_n_quad `"`e(n_quad)'"'
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ///
    dynamic_random_effects_ordered_logit
predict double p0 p1 p2, pr conditional(fixedonly)
ldk_post_predictions `prediction_handle', ///
    model(dynamic_random_effects_ordered_logit) id(obs_id) ///
    probabilities(p0 p1 p2)
drop p0 p1 p2

postclose `coefficient_handle'
postclose `covariance_handle'
postclose `fit_handle'
postclose `prediction_handle'

use `"`coefficient_data'"', clear
sort model position
format estimate standard_error %24.17g
export delimited using `"`output_dir'/estimates_raw.csv"', replace datafmt

use `"`covariance_data'"', clear
sort model row_position column_position
format covariance %24.17g
export delimited using `"`output_dir'/covariance_raw.csv"', replace datafmt

use `"`fit_data'"', clear
sort model
format nobs n_groups n_params loglike aic bic converged %24.17g
export delimited using `"`output_dir'/fit.csv"', replace datafmt

use `"`prediction_data'"', clear
sort model obs_id category
format probability %24.17g
export delimited using `"`output_dir'/predictions.csv"', replace datafmt

file open `metadata_handle' using `"`output_dir'/metadata.txt"', write text replace
file write `metadata_handle' "suite=real_data_application" _n
file write `metadata_handle' "stata_version=`c(stata_version)'" _n
file write `metadata_handle' "gologit2_installed=`have_gologit2'" _n
file write `metadata_handle' `"gologit2_path=`gologit2_path'"' _n
file write `metadata_handle' ///
    `"random_effects_ordered_logit.intmethod=`static_re_intmethod'"' _n
file write `metadata_handle' ///
    `"random_effects_ordered_logit.n_quad=`static_re_n_quad'"' _n
file write `metadata_handle' ///
    `"dynamic_random_effects_ordered_logit.intmethod=`dynamic_re_intmethod'"' _n
file write `metadata_handle' ///
    `"dynamic_random_effects_ordered_logit.n_quad=`dynamic_re_n_quad'"' _n
file write `metadata_handle' "panel_prediction=conditional_fixedonly" _n
file write `metadata_handle' "source_release=Stata Press r19" _n
file write `metadata_handle' "run_completed=1" _n
file close `metadata_handle'

display as result "Stata real-data parity outputs written to `output_dir'"
log close
