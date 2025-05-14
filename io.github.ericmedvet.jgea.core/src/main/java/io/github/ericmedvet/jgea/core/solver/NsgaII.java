/*-
 * ========================LICENSE_START=================================
 * jgea-core
 * %%
 * Copyright (C) 2018 - 2024 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */
package io.github.ericmedvet.jgea.core.solver;

import io.github.ericmedvet.jgea.core.Factory;
import io.github.ericmedvet.jgea.core.operator.GeneticOperator;
import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.order.PartiallyOrderedCollection;
import io.github.ericmedvet.jgea.core.problem.MultiObjectiveProblem;
import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

// source -> https://doi.org/10.1109/4235.996017

public class NsgaII<G, S, Q> extends AbstractPopulationBasedIterativeSolver<POCPopulationState<Individual<G, S, Q>, G, S, Q, MultiObjectiveProblem<S, Q, Double>>, MultiObjectiveProblem<S, Q, Double>, Individual<G, S, Q>, G, S, Q> {

  protected final Map<GeneticOperator<G>, Double> operators;
  private final int populationSize;
  private final int maxUniquenessAttempts;

  public NsgaII(
      Function<? super G, ? extends S> solutionMapper,
      Factory<? extends G> genotypeFactory,
      int populationSize,
      Predicate<? super POCPopulationState<Individual<G, S, Q>, G, S, Q, MultiObjectiveProblem<S, Q, Double>>> stopCondition,
      Map<GeneticOperator<G>, Double> operators,
      int maxUniquenessAttempts,
      boolean remap,
      List<PartialComparator<? super Individual<G, S, Q>>> additionalIndividualComparators
  ) {
    super(solutionMapper, genotypeFactory, stopCondition, remap, additionalIndividualComparators);
    this.operators = operators;
    this.populationSize = populationSize;
    this.maxUniquenessAttempts = maxUniquenessAttempts;
  }

  private record RankedIndividual<G, S, Q>(
      long id,
      G genotype,
      S solution,
      Q quality,
      long qualityMappingIteration,
      long genotypeBirthIteration,
      int rank,
      double crowdingDistance,
      Collection<Long> parentIds
  ) implements Individual<G, S, Q> {
    static <G, S, Q> RankedIndividual<G, S, Q> from(
        Individual<G, S, Q> individual
    ) {
      return new RankedIndividual<>(
          individual.id(),
          individual.genotype(),
          individual.solution(),
          individual.quality(),
          individual.qualityMappingIteration(),
          individual.genotypeBirthIteration(),
          0,
          0,
          individual.parentIds()
      );
    }
  }

  private static <G, S, Q> Collection<RankedIndividual<G, S, Q>> decorate(
      Collection<? extends Individual<G, S, Q>> individuals,
      MultiObjectiveProblem<S, Q, Double> problem,
      PartialComparator<? super Individual<G, S, Q>> partialComparator
  ) {
    List<? extends Collection<? extends Individual<G, S, Q>>> fronts = PartiallyOrderedCollection
        .from(
            individuals,
            partialComparator
        )
        .fronts();
    SequencedMap<String, MultiObjectiveProblem.Objective<Q, Double>> objectives = problem.objectives();
    return IntStream.range(0, fronts.size())
        .mapToObj(fi -> {
          List<? extends Individual<G, S, Q>> is = fronts.get(fi)
              .stream()
              .toList();
          List<Double> distances = distances(is, objectives);
          return IntStream.range(0, is.size())
              .mapToObj(ii -> {
                Individual<G, S, Q> individual = is.get(ii);
                return new RankedIndividual<>(
                    individual.id(),
                    individual.genotype(),
                    individual.solution(),
                    individual.quality(),
                    individual.qualityMappingIteration(),
                    individual.genotypeBirthIteration(),
                    fi,
                    distances.get(ii),
                    individual.parentIds()
                );
              })
              .toList();
        })
        .flatMap(List::stream)
        .toList();
  }

  private static <G, S, Q> List<Double> distances(
      List<? extends Individual<G, S, Q>> individuals,
      SequencedMap<String, MultiObjectiveProblem.Objective<Q, Double>> objectives
  ) {
    double[] dists = new double[individuals.size()];
    for (MultiObjectiveProblem.Objective<Q, Double> objective : objectives.values()) {
      List<Integer> indexes = IntStream.range(0, individuals.size())
          .boxed()
          .sorted(
              Comparator.comparing(
                  ii -> objective.function().apply(individuals.get(ii).quality()),
                  objective.comparator()
              )
          )
          .toList();
      for (int ii = 1; ii < indexes.size() - 1; ii = ii + 1) {
        int previousIndex = indexes.get(ii - 1);
        int nextIndex = indexes.get(ii + 1);
        double dist = Math.abs(
            objective.function().apply(individuals.get(previousIndex).quality()) - objective.function()
                .apply(individuals.get(nextIndex).quality())
        );
        dists[indexes.get(ii)] = dists[indexes.get(ii)] + dist;
      }
      dists[indexes.getFirst()] = dists[indexes.getFirst()] + Double.POSITIVE_INFINITY;
      dists[indexes.getLast()] = dists[indexes.getLast()] + Double.POSITIVE_INFINITY;
    }
    return Arrays.stream(dists).boxed().toList();
  }

  private static <G, S, Q> Comparator<RankedIndividual<G, S, Q>> rankedComparator() {
    return Comparator.comparingInt((RankedIndividual<G, S, Q> i) -> i.rank)
        .thenComparing((i1, i2) -> Double.compare(i2.crowdingDistance, i1.crowdingDistance));
  }

  @Override
  public POCPopulationState<Individual<G, S, Q>, G, S, Q, MultiObjectiveProblem<S, Q, Double>> init(
      MultiObjectiveProblem<S, Q, Double> problem,
      RandomGenerator random,
      ExecutorService executor
  ) throws SolverException {
    POCPopulationState<Individual<G, S, Q>, G, S, Q, MultiObjectiveProblem<S, Q, Double>> newState = POCPopulationState
        .empty(problem, stopCondition());
    AtomicLong counter = new AtomicLong(0);
    List<? extends G> genotypes = genotypeFactory.build(populationSize, random);
    Collection<Individual<G, S, Q>> individuals = getAll(
        map(
            genotypes.stream()
                .map(g -> new ChildGenotype<G>(counter.getAndIncrement(), g, List.of()))
                .toList(),
            (cg, s, r) -> RankedIndividual.from(
                Individual.from(cg, solutionMapper, s.problem().qualityFunction(), s.nOfIterations())
            ),
            newState,
            random,
            executor
        )
    );
    return newState.updatedWithIteration(
        genotypes.size(),
        genotypes.size(),
        PartiallyOrderedCollection.from(individuals, partialComparator(problem))
    );
  }

  @Override
  public POCPopulationState<Individual<G, S, Q>, G, S, Q, MultiObjectiveProblem<S, Q, Double>> update(
      RandomGenerator random,
      ExecutorService executor,
      POCPopulationState<Individual<G, S, Q>, G, S, Q, MultiObjectiveProblem<S, Q, Double>> state
  ) throws SolverException {
    // build offspring
    Collection<ChildGenotype<G>> offspringChildGenotypes = new ArrayList<>();
    Set<G> uniqueOffspringGenotypes = new HashSet<>();
    if (maxUniquenessAttempts > 0) {
      uniqueOffspringGenotypes.addAll(
          state.pocPopulation()
              .all()
              .stream()
              .map(Individual::genotype)
              .toList()
      );
    }
    int attempts = 0;
    List<Individual<G, S, Q>> individuals = state.pocPopulation()
        .all()
        .stream()
        .toList();
    int size = individuals.size();
    AtomicLong counter = new AtomicLong(state.nOfBirths());
    while (offspringChildGenotypes.size() < populationSize) {
      GeneticOperator<G> operator = Misc.pickRandomly(operators, random);
      List<Individual<G, S, Q>> parents = IntStream.range(
          0,
          operator.arity()
      )
          .mapToObj(n -> individuals.get(Math.min(random.nextInt(size), random.nextInt(size))))
          .toList();
      List<? extends G> newGenotypes = operator.apply(parents.stream().map(Individual::genotype).toList(), random);
      if (attempts >= maxUniquenessAttempts || newGenotypes.stream().noneMatch(uniqueOffspringGenotypes::contains)) {
        attempts = 0;
        List<Long> parentIds = parents.stream().map(Individual::id).toList();
        newGenotypes.forEach(
            g -> offspringChildGenotypes.add(new ChildGenotype<>(counter.getAndIncrement(), g, parentIds))
        );
        uniqueOffspringGenotypes.addAll(newGenotypes);
      } else {
        attempts = attempts + 1;
      }
    }
    // map and decorate and trim
    Collection<Individual<G, S, Q>> newPopulation = mapAll(
        offspringChildGenotypes,
        (cg, s, r) -> RankedIndividual.from(
            Individual.from(cg, solutionMapper, s.problem().qualityFunction(), s.nOfIterations())
        ),
        state.pocPopulation().all(),
        (i, s, r) -> RankedIndividual.from(i.updatedWithQuality(s)),
        state,
        random,
        executor
    );
    PartialComparator<? super Individual<G, S, Q>> partialComparator = partialComparator(state.problem());
    List<RankedIndividual<G, S, Q>> rankedIndividuals = decorate(newPopulation, state.problem(), partialComparator)
        .stream()
        .sorted(rankedComparator())
        .limit(populationSize)
        .toList();
    @SuppressWarnings({"unchecked", "rawtypes"}) List<Individual<G, S, Q>> newIndividuals = (List) rankedIndividuals;
    return state.updatedWithIteration(
        offspringChildGenotypes.size(),
        offspringChildGenotypes.size() + (remap ? populationSize : 0),
        PartiallyOrderedCollection.from(newIndividuals, partialComparator)
    );
  }
}
