This is was a final project that I chose to submit for the class 
ECON 21110 (Applied Microeconometrics) at UChicago. My analysis replicates 
key results from Maestas et al. (2023)

Paper Summary:
The goal of the paper is to estimate the effect of aging on GDP growth. The 
central challenge of this question is that the economic growth of a given state 
and the fraction of its population that is 60+ do not have a clear causal relationship: 
prime-aged workers may decide to leave a state due to some negative economic shock, 
making it appear as though population aging causes slower economic growth when in 
fact the reverse relationship is true. Therefore, to address simultaneity, the paper 
employs an instrumental variables (IV) approach to estimate how a change in fraction of 60+ 
population impacts GDP.

Specific Findings:
- Every 10% increase in the fraction of 60+ pop. decreased GDP by 5.5%
    - 1/3 of this came from slower employment growth
    - 2/3 came from slower labor productivity growth
- The above implies population aging reduced growth in GDP by 0.3% from 1980-2010
- Authors estimate between 2010-2020 a projected rise in population share aged 60+ 
by around 21% could slow GDP per capita growth by 1.2% annually, or a 0.6% annual 
slowdown between 2020-2030. These results are larger than National Research 
projection of 0.33-0.55 ppt slowdown because their model only accounts for 
labor force decline

Paper Limitations:
 - generalizability from states to nation (state-based research designes avoid
  capturing any federal policy responses that accrue uniformly across states)







Notation:
- ln(y_s,t): log of GDP per capita in state s at year t
- ∆ln(y_s,t): Growth rate in GDP per capita between year t and t + 10
- A_s,t: # ppl aged 60+ in state s at year t
- N_s,t: # ppl aged 20 and older in state s at year t
- ∆ln(A / N)_s,t: Growth in share of ppl aged 60+ in state s at year t

- epsilon_s,t: unobserved shocks to GDP per capita in state s and year t
- ∆epsilon_s,t: change in unobservable shocks between year t and t + 10

IVs:
Modeling shocks is difficult, but some of the observed variation 
in population aging across states was determined many years before. Under 
certain conditions, we can use this predetermined component as an instrumental
variable for the realized aging experienced by a state many years later. Our key 
assumption here is that a state's past age structure affects its future economic 
position only by affecting its subsequently realized age structure. 

- A^hat_s,t+10: predicted 60+ population in state s and year t + 10 using past 
    age structure and national survival rates
- N^hat_s,t+10: predicted 20+ population
- ∆ln(A^hat / N^hat)_s,t: Growth in predicted 60+ share <-- instrument
- pi: first-stage coefficient





Basic setup:
Y = beta * X + error
- Y = beta * X + epsilon
- Y: ∆ in GDP/capita growth
- X: ∆ in old ppl %

Z: predicted growth in age structure of a state, build from a state age structure 
lagged at p /in {10, 20, 30, 40} years prior and aged forward using national growth rates