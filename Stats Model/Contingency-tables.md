# Contingency Tables

Analysis a 2x2 contingent table with counts below:

| Gender\event  |       Yes     | No    | Total |
| ------------- |:-------------:| -----:|------:|
| Female        | A             | B     | A+B   |
| Male          | C             | D     | C+D   |
| Total         | A+C           | B+D   |A+C+B+D|

## Three probs:
- Joint probabilities: the prob events occur together

  - example: what is a joint probablit of disease diagnosed by a female? A/(A+B+C+D) %
  - formula: P(F inter Y)=count in (F,Y) / grand total
  - the sum of joint probablities for the entire table sum to 1.

- Marginal prob:

prob that a signle event occurs with no regard to other events in the table (prob do not depend on the condition of another outcome).

   - P(female)=A+B/(A+B+C+D)
   - P(Yes)=A+C/(A+B+C+D)

   - column/row marginal prob sum to 1.

- Conditional prob:

The prob that an event occurs given that another event has occured. 

   - ex: given a gender is female, what is the the prob she is sick? P(Y|F)=A/(A+B) %
   - ex: given a yes, what is the prob that is the male? P(M|Y)=C/(A+C) %





