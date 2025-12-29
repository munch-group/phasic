



Is there a "moments_discrete"


compute_moments_impl cpp function does not compute discrete moments i think


the * versions measure nr time, and rewards accumulated

the *_discrete versions measure nr jumps, and visits



please explain how method expectation differs from expectation_discrete, how pdf differs from pdf_discrete, and how all similar pairs of continous and discrete methods differ



CHECK HOW SAMPLE_DISCRETE WORKS

what does normalize_discrete do?

Consider removing:
expected_residence_time
expected_visits
accumulated_visiting_time
accumulated_visits




For discrete popgen models I need to be able to scale N and R with mutation rate.
If mut rate is a a free param, it is is confounded with N

coal rate: [pairs, 0, 0] * 1/N  -> [pairs/N, 0, 0] 
rec rate:  [0, lins, 0] * R     -> [0, lins*R, 0] 
mut rate:  [0, 0, 1] * u        -> [0, 0, 1]      



need to sample sample_multivariate, which gets me one t-state at a time, and then map back to the actual rewards



you are working on the joint prob sampling on the simpler coalescent ...