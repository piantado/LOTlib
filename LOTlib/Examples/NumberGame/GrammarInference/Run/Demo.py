from Run import *
from LOTlib.Grammar import Grammar
from LOTlib.DataAndObjects import FunctionData

# --------------------------------------------------------------------------------------------------------
# Mixture model

if __name__ == "__main__":

    path = os.getcwd()

    mix_grammar_test = Grammar()

    mix_grammar_test.add_rule('START', '', ['MATH'], 1.)
    mix_grammar_test.add_rule('MATH', 'mapset_', ['FUNC', 'DOMAIN_RANGE'], 1.)
    mix_grammar_test.add_rule('DOMAIN_RANGE', 'range_set_', ['1', '100'], 1.)
    mix_grammar_test.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='X', bv_p=1.)
    mix_grammar_test.add_rule('EXPR', 'ipowf_', [str(2), 'X'], 1.)

    mix_grammar_test.add_rule('START', '', ['INTERVAL'], 1.)
    mix_grammar_test.add_rule('INTERVAL', 'range_set_', ['1', '100'], 1.)

    interval_data = [FunctionData(
        input=[16],
        output={99: (30, 5), 64: (5, 30)})]

    math_data = [FunctionData(
        input=[16],
        output={99: (5, 30), 64: (30, 5)})]

    run(grammar=mix_grammar, mixture_model=1, data=math_data,
        ngh='enum7', domain=100, alpha=0.9,
        iters=120000, skip=120, cap=1000,
        print_stuff='', pickle_file='out/mix_math_120k.p',
        csv_file=path+'/out/mix_math_120k')

    # --------------------------------------------------------------------------------------------------------
    # Individual rule probabilities model

    # run(grammar=independent_grammar, mixture_model=0, data=toy_3n,
    #     ngh='enum7', domain=100, alpha=0.9,
    #     iters=10000, skip=10, cap=1000,
    #     print_stuff='rs', pickle_file='',
    #     csv_file=path+'/out/indep_toy3n_10k')

    # --------------------------------------------------------------------------------------------------------
    # LOT grammar

    # run(grammar=lot_grammar, mixture_model=0, data=josh_data, domain=100, alpha=0.9,
    #     ngh='load', ngh_file=path+'/ngh_mcmc100k.p', grammar_n=1000000, skip=200, cap=5000,
    #     print_stuff='', plot_type='', gh_pickle='save',
    #     gh_file=path+'/out/2_2/lot_1mil_1.p',
    #     csv_save=path+'/out/2_2/lot_1mil_1')

    # --------------------------------------------------------------------------------------------------------
    # TESTING

    # run(grammar=complex_grammar, data=toy_2n, domain=20,
    #     alpha=0.99, ngh=6, grammar_n=1000, skip=10, cap=100,
    #     print_stuff='rules', plot_type=[], gh_pickle='save',
    #     gh_file='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference/NumberGame'
    #              '/out/newest_complex_2n_1000.p')

    # --------------------------------------------------------------------------------------------------------

