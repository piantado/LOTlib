from Run import *

# --------------------------------------------------------------------------------------------------------
# Mixture model

# import cProfile
# cProfile.run("""run(grammar=mix_grammar, josh='mix', data=josh_data, domain=100,
#                     alpha=0.9, ngh=5, grammar_n=50, skip=10, cap=100,
#                     print_stuff=[], plot_type=[], gh_pickle=False)""",
#              filename=path+'/out/profile/mix_model_50.profile')

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
    output={99: (30, 0), 64: (0, 30)})]

math_data = [FunctionData(
    input=[16],
    output={99: (0, 30), 64: (30, 0)})]

run(grammar=mix_grammar_test, mixture_model=1, data=math_data,
    ngh='enum7', iters=1000, skip=10, cap=100,
    print_stuff='s', plot_type='', pickle_file='',  # gh_file=path+'/out/2_4/mix_math_1k.p',
    csv_file=path+'/out/2_4/mix_math_1k')

# --------------------------------------------------------------------------------------------------------
# Individual rule probabilities model

# run(grammar=individual_grammar, mixture_model=0, data=josh_data, domain=100, alpha=0.9,
#     ngh='enum7', grammar_n=1000000, skip=200, cap=5000,
#     print_stuff='', plot_type='', gh_pickle='save',
#     gh_file=path+'/out/2_2/indep_1mil_1.p',
#     csv_save=path+'/out/2_2/indep_1mil_1')

# --------------------------------------------------------------------------------------------------------
# LOT grammar

# run(grammar=lot_grammar, mixture_model=0, data=josh_data, domain=100, alpha=0.9,
#     ngh='load', ngh_file=path+'/ngh_mcmc100k.p', grammar_n=1000000, skip=200, cap=5000,
#     print_stuff='', plot_type='', gh_pickle='save',
#     gh_file=path+'/out/2_2/lot_1mil_1.p',
#     csv_save=path+'/out/2_2/lot_1mil_1')

# --------------------------------------------------------------------------------------------------------
# TESTING  |  Original number game

# run(grammar=complex_grammar, data=toy_2n, domain=20,
#     alpha=0.99, ngh=6, grammar_n=1000, skip=10, cap=100,
#     print_stuff='rules', plot_type=[], gh_pickle='save',
#     gh_file='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference/NumberGame'
#              '/out/newest_complex_2n_1000.p')

# --------------------------------------------------------------------------------------------------------


#
#
# '''print distribution over power rule:  [prior, likelihood, posterior]'''
# # vals, posteriors = grammar_h0.rule_distribution(data, 'ipowf_', np.arange(0.1, 5., 0.1))
# # print_dist(vals, posdfteriors)
# #visualize_dist(vals, posteriors, 'union_')

