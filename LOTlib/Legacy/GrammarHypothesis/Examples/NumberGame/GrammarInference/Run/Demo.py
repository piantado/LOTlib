from Run import *
from LOTlib.Grammar import Grammar
from LOTlib.DataAndObjects import FunctionData

# --------------------------------------------------------------------------------------------------------
# Mixture model

if __name__ == "__main__":

    path = os.getcwd()

    interval_data = [FunctionData(
        input=[16],
        output={99: (30, 5), 64: (5, 30)})]

    math_data = [FunctionData(
        input=[16],
        output={99: (5, 30), 64: (30, 5)})]

    # run(grammar=mix_grammar, mixture_model=1, data=math_data,
    #     ngh='enum7', domain=100, alpha=0.9,
    #     iters=120000, skip=120, cap=1000,
    #     print_stuff='', pickle_file='out/mix_math_120k.p',
    #     csv_file=path+'/out/mix_math_120k')

    grammar_n = 10000
    skip = 10
    cap = grammar_n/skip

    hypotheses = []
    for fn in mix_grammar.enumerate(d=6):
        h = NumberGameHypothesis(grammar=mix_grammar, domain=100, alpha=0.9)
        h.set_value(fn)
        hypotheses.append(h)

    # 16 => 15
    BDA_data_similarity_1 = [
        FunctionData(
            input=[16],
            output={15: (3, 3)})]
    BDA_data_similarity_2 = [
        FunctionData(
            input=[16],
            output={15: (30, 3)})]
    BDA_data_similarity_3 = [
        FunctionData(
            input=[16],
            output={15: (3, 30)})]
    BDA_data_similarity_4 = [
        FunctionData(
            input=[16],
            output={15: (30, 30)})]

    # 4 => 99
    BDA_data_rule_1 = [
        FunctionData(
            input=[16],
            output={64: (3, 3)})]
    BDA_data_rule_2 = [
        FunctionData(
            input=[16],
            output={64: (30, 3)})]
    BDA_data_rule_3 = [
        FunctionData(
            input=[16],
            output={64: (3, 30)})]
    BDA_data_rule_4 = [
        FunctionData(
            input=[16],
            output={64: (30, 30)})]

    mixture_ratios = [None]*8
    for i,d in enumerate((BDA_data_similarity_1, BDA_data_similarity_2, BDA_data_similarity_3, BDA_data_similarity_4,
                          BDA_data_rule_1, BDA_data_rule_2, BDA_data_rule_3, BDA_data_rule_4)):
        grammar_h0 = MixtureGrammarHypothesis(mix_grammar, hypotheses, propose_scale=.1, propose_n=1)
        mh_grammar_sampler = MHSampler(grammar_h0, d, grammar_n)
        mh_grammar_summary = VectorSummary(skip=skip, cap=cap)

        for h in mh_grammar_summary(mh_grammar_sampler):
            pass

        # mixture_ratios[i] = [(s.value[0]/(s.value[0]+s.value[1])) for s in mh_grammar_summary.samples]
        mixture_ratios[i] = [(s.value[s.get_rules(rule_to='MATH')[0][0]] / (s.value[s.get_rules(rule_to='MATH')[0][0]]+s.value[s.get_rules(rule_to='INTERVAL')[0][0]])) for s in mh_grammar_summary.samples]

    f = open("out/mix_ratios.p", "wb")
    pickle.dump(mixture_ratios, f)



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

