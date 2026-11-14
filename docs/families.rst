Lag-0 explanation families
==========================

With coarse sampling, processes interact within one sample, and time order no longer orients
those links. The PC-style lag-0 layer (:func:`infoflow.contemporaneous.contemporaneous_layer`)
reads adjacencies from dependence :cite:`Runge2020`. That assumes faithfulness
:cite:`Spirtes2000`: every independence in the data comes from a separation in the causal
graph. Synergy breaks it. If :math:`Z_t = X_t \oplus Y_t \oplus E_t`, every pair of the three
is independent, so the layer reports no link at all, although :math:`X_t` and :math:`Y_t`
together determine :math:`Z_t`.

``infer_multiplex(contemporaneous="family")``
(:func:`infoflow.contemporaneous.contemporaneous_families`) instead reports every explanation
that *weaker adjacency faithfulness* allows :cite:`Marx2021weaker,Hourican2026`. A link may
then be supported by a triple association, as in XOR, and the answer is a family of DAGs. For
the XOR above it is a collider at each of the three variables. Neither a single skeleton nor
the union of the three (a triangle) represents that answer.

.. code-block:: python

   net = infoflow.infer_multiplex(data, names=["X", "Y", "Z"], contemporaneous="family", prng=0)
   for family in net.contemporaneous_families:
       print(family.describe(net.names))
   # {X, Y, Z}: 3 explanations: {X -> Y, Z -> Y} | {X -> Z, Y -> Z} | {Y -> X, Z -> X}

Lagged links need no families: time order already leaves one explanation for them, and the
lagged search finds jointly informative parents with its pair search (:doc:`skeleton`).

Procedure
---------

1. **Screen.** Two processes are linked when their present values are dependent given the
   lagged conditioning set, or given that set and the present value of one more process.
   The lagged conditioning set holds both processes' selected pasts and lagged parents, as in
   the PC-style layer. Linked processes form components.
2. **Decisions.** Each component of at most five processes is decided statement by statement.
   Every pairwise statement :math:`X^i_t \perp X^j_t \mid X^S_t`, with :math:`S` a set of the
   other processes in the component, is tested given the pair's lagged conditioning set. The
   level is :math:`\alpha / m` for the component's :math:`m` statements (6 for three
   processes, 24 for four, 80 for five).
3. **Family.** Every DAG on the component (25, 543 or 29,281 of them) is kept when it is

   * *Markov*: every statement it implies by d-separation was decided independent;
   * *weakly adjacency faithful*: at each endpoint :math:`x` of each edge :math:`x - w`, either
     :math:`x` and :math:`w` are dependent given every conditioning set, or they form a strict
     two-association with a third member :math:`z` of :math:`x`'s Markov blanket. Each of the
     three pairs is then dependent given the remaining variable and any further set, while
     neither :math:`x, w` nor :math:`x, z` is dependent given every set
     (``convention="neither"``; ``"not_both"`` only requires that not both are);
   * *SGS-minimal*: no DAG with one edge fewer is Markov;
   * consistent with background knowledge: ``tiers`` (a known order within a sample) and
     ``forbidden`` arcs, passed through ``contemporaneous_settings``.

Reading the result
------------------

``network.contemporaneous_families`` lists one :class:`~infoflow.families.ExplanationFamily`
per component, with its explanations and a status:

``complete``
   Every DAG was checked, and these are all that qualify.
``empty``
   None qualifies: the assumptions fail for these processes, or the decisions contradict each
   other.
``too_large``
   The component has more than ``max_family_size`` processes.
``unavailable``
   A statement's table has fewer than ``min_cell_count`` samples per cell on average, so not
   rejecting it would be no evidence of independence.

The last two keep the PC-style layer's entries and make no further claim.

``dataset["contemporaneous"]`` marks only what every explanation shares: 2 for an arrow in
all of them, 1 for a link in all of them whose direction varies, 0 otherwise. A lag-0 XOR
therefore gives an all-zero matrix. ``dataset["contemporaneous_possible"]`` marks the links
that appear in at least one explanation.

Why the decisions are corrected
-------------------------------

A true DAG implies many independences, and one false rejection among them removes it from
the family. At an uncorrected level of 0.05 per statement, families are therefore often
empty, which is the finite-sample problem :cite:`Hourican2026` report for their family
estimator. We simulated four lag-0 motifs (XOR, XOR with a descendant, OR, and a
four-process chain) at 2000 and 10,000 samples:

* Uncorrected decisions returned exactly the population family in 55--90% of samples, and
  an empty family in most of the rest.
* With the Bonferroni level, asymptotic tests returned it in all 20 samples of every motif.
  Permutation tests (the default) returned it in all 20 samples of every motif at 2000
  samples.

The price is power for weak lag-0 dependence.

Assumptions and limits
----------------------

The explanations assume:

* no hidden common causes of the measured processes within a sample;
* acyclic effects within a sample;
* no interaction between components;
* lag-0 permutation tests that are adequate despite serial dependence (they condition on the
  lagged pasts).

Components of more than five processes are not enumerated.

:func:`infoflow.families.decisions_from_distribution` gives exact decisions for a known
distribution. With it, :func:`infoflow.families.explanation_family` computes the population
families that the tests are checked against.
