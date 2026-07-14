version 15.1
clear all
set more off
set linesize 255

/*
Run from Stata with the work directory produced by prepare_parity.py:

    do "validation/stata/limiteddepkit_parity.do" ///
        "C:/path/to/limiteddepkit/validation/stata/work"

The script uses only official Stata commands except for the two optional
gologit2 fits. It never installs community commands automatically.
*/

args workdir
if `"`workdir'"' == "" {
    display as error "Pass the parity work directory as the first argument."
    exit 198
}

local data_dir `"`workdir'/data"'
local output_dir `"`workdir'/stata"'
capture mkdir `"`output_dir'"'

foreach fixture in cross_section static_re dynamic_design {
    capture confirm file `"`data_dir'/`fixture'.dta"'
    if _rc {
        display as error "Missing `data_dir'/`fixture'.dta"
        display as error "Run prepare_parity.py before this do-file."
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

display as text "Running pooled binary and ordinal parity fits"
use `"`data_dir'/cross_section.dta"', clear

quietly logit y_logit intercept x1 x2, noconstant vce(oim)
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' binary_logit
predict double p1, pr
generate double p0 = 1 - p1
ldk_post_predictions `prediction_handle', model(binary_logit) id(obs_id) ///
    probabilities(p0 p1)
drop p0 p1

quietly probit y_probit intercept x1 x2, noconstant vce(oim)
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' binary_probit
predict double p1, pr
generate double p0 = 1 - p1
ldk_post_predictions `prediction_handle', model(binary_probit) id(obs_id) ///
    probabilities(p0 p1)
drop p0 p1

quietly ologit y_ologit ox1 ox2, vce(oim)
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ordered_logit
predict double p0, outcome(#1)
predict double p1, outcome(#2)
predict double p2, outcome(#3)
ldk_post_predictions `prediction_handle', model(ordered_logit) id(obs_id) ///
    probabilities(p0 p1 p2)
drop p0 p1 p2

quietly oprobit y_oprobit ox1 ox2, vce(oim)
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ordered_probit
predict double p0, outcome(#1)
predict double p1, outcome(#2)
predict double p2, outcome(#3)
ldk_post_predictions `prediction_handle', model(ordered_probit) id(obs_id) ///
    probabilities(p0 p1 p2)
drop p0 p1 p2

capture which gologit2
local have_gologit2 = (_rc == 0)
if `have_gologit2' {
    display as text "Running optional gologit2 parity fits"
    which gologit2

    quietly gologit2 y_gologit gx1 gx2, npl
    ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ///
        generalized_ordered_logit
    predict double p0 p1 p2
    ldk_post_predictions `prediction_handle', model(generalized_ordered_logit) ///
        id(obs_id) probabilities(p0 p1 p2)
    drop p0 p1 p2

    quietly gologit2 y_gologit gx1 gx2, npl(gx1)
    ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ///
        partial_proportional_odds
    predict double p0 p1 p2
    ldk_post_predictions `prediction_handle', model(partial_proportional_odds) ///
        id(obs_id) probabilities(p0 p1 p2)
    drop p0 p1 p2
}
else {
    display as result "gologit2 is not installed; flexible ordinal fits were skipped."
    display as result "Install it manually with: ssc install gologit2"
}

display as text "Running static random-effects Ordered Logit parity fit"
use `"`data_dir'/static_re.dta"', clear
quietly meologit y x1 x2 || entity:, intmethod(ghermite) intpoints(12) vce(oim) ///
    iterate(2000) tolerance(1e-10) ltolerance(1e-12) nrtolerance(1e-8)
local static_re_intmethod `"`e(intmethod)'"'
local static_re_n_quad `"`e(n_quad)'"'
ldk_post_current `coefficient_handle' `covariance_handle' `fit_handle' ///
    random_effects_ordered_logit
predict double p0 p1 p2, pr conditional(fixedonly)
ldk_post_predictions `prediction_handle', model(random_effects_ordered_logit) ///
    id(obs_id) probabilities(p0 p1 p2)
drop p0 p1 p2

display as text "Running dynamic augmented-design Ordered Logit parity fit"
use `"`data_dir'/dynamic_design.dta"', clear
quietly meologit y x1 state_1 state_2 initial_1 initial_2 initial_x1 mean_x1 ///
    || entity:, intmethod(ghermite) intpoints(12) vce(oim) ///
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
file write `metadata_handle' "suite=controlled_synthetic_certification" _n
file write `metadata_handle' "stata_version=`c(stata_version)'" _n
file write `metadata_handle' "gologit2_installed=`have_gologit2'" _n
file write `metadata_handle' ///
    `"random_effects_ordered_logit.intmethod=`static_re_intmethod'"' _n
file write `metadata_handle' ///
    `"random_effects_ordered_logit.n_quad=`static_re_n_quad'"' _n
file write `metadata_handle' ///
    `"dynamic_random_effects_ordered_logit.intmethod=`dynamic_re_intmethod'"' _n
file write `metadata_handle' ///
    `"dynamic_random_effects_ordered_logit.n_quad=`dynamic_re_n_quad'"' _n
file write `metadata_handle' "panel_prediction=conditional_fixedonly" _n
file write `metadata_handle' "run_completed=1" _n
file close `metadata_handle'

display as result "Stata parity outputs written to `output_dir'"
log close
