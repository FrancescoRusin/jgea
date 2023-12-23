/*-
 * ========================LICENSE_START=================================
 * jgea-core
 * %%
 * Copyright (C) 2018 - 2023 Eric Medvet
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
import io.github.ericmedvet.jgea.core.problem.MultiHomogeneousObjectiveProblem;
import io.github.ericmedvet.jgea.core.util.Misc;
import io.github.ericmedvet.jgea.core.util.Progress;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

// source -> https://doi.org/10.1109/4235.996017

public class NsgaII<G, S>
    extends AbstractPopulationBasedIterativeSolver<
        POCPopulationState<Individual<G, S, List<Double>>, G, S, List<Double>>,
        MultiHomogeneousObjectiveProblem<S, Double>,
        Individual<G, S, List<Double>>,
        G,
        S,
        List<Double>> {

  protected final Map<GeneticOperator<G>, Double> operators;
  private final int populationSize;
  private final int maxUniquenessAttempts;

  public NsgaII(
      Function<? super G, ? extends S> solutionMapper,
      Factory<? extends G> genotypeFactory,
      int populationSize,
      Predicate<? super POCPopulationState<Individual<G, S, List<Double>>, G, S, List<Double>>> stopCondition,
      Map<GeneticOperator<G>, Double> operators,
      int maxUniquenessAttempts,
      boolean remap) {
    super(solutionMapper, genotypeFactory, stopCondition, remap);
    this.operators = operators;
    this.populationSize = populationSize;
    this.maxUniquenessAttempts = maxUniquenessAttempts;
  }

  private record RankedIndividual<G, S>(
      G genotype,
      S solution,
      List<Double> quality,
      long qualityMappingIteration,
      long genotypeBirthIteration,
      int rank,
      double crowdingDistance)
      implements Individual<G, S, List<Double>> {}

  private record State<G, S>(
      LocalDateTime startingDateTime,
      long elapsedMillis,
      long nOfIterations,
      Progress progress,
      long nOfBirths,
      long nOfFitnessEvaluations,
      PartiallyOrderedCollection<Individual<G, S, List<Double>>> pocPopulation,
      List<RankedIndividual<G, S>> listPopulation)
      implements POCPopulationState<Individual<G, S, List<Double>>, G, S, List<Double>> {
    public static <G, S> State<G, S> from(
        State<G, S> state,
        Progress progress,
        long nOfBirths,
        long nOfFitnessEvaluations,
        List<RankedIndividual<G, S>> listPopulation,
        PartialComparator<? super Individual<G, S, List<Double>>> partialComparator) {
      //noinspection rawtypes,unchecked
      return new State<>(
          state.startingDateTime,
          ChronoUnit.MILLIS.between(state.startingDateTime, LocalDateTime.now()),
          state.nOfIterations() + 1,
          progress,
          state.nOfBirths() + nOfBirths,
          state.nOfFitnessEvaluations() + nOfFitnessEvaluations,
          PartiallyOrderedCollection.from((Collection) listPopulation, partialComparator),
          listPopulation);
    }

    public static <G, S> State<G, S> from(
        List<RankedIndividual<G, S>> listPopulation,
        PartialComparator<? super Individual<G, S, List<Double>>> partialComparator) {
      //noinspection rawtypes,unchecked
      return new State<>(
          LocalDateTime.now(),
          0,
          0,
          Progress.NA,
          listPopulation.size(),
          listPopulation.size(),
          PartiallyOrderedCollection.from((Collection) listPopulation, partialComparator),
          listPopulation);
    }
  }

  private static <G, S> Comparator<RankedIndividual<G, S>> rankedComparator() {
    return Comparator.comparingInt((RankedIndividual<G, S> i) -> i.rank)
        .thenComparing((i1, i2) -> Double.compare(i2.crowdingDistance, i1.crowdingDistance));
  }

  private static <G, S> List<Double> distances(
      List<? extends Individual<G, S, List<Double>>> individuals, List<Comparator<Double>> comparators) {
    double[] dists = new double[individuals.size()];
    for (int oI = 0; oI < comparators.size(); oI = oI + 1) {
      int finalOI = oI;
      List<Integer> indexes = IntStream.range(0, individuals.size())
          .boxed()
          .sorted((ii1, ii2) -> comparators
              .get(finalOI)
              .compare(
                  individuals.get(ii1).quality().get(finalOI),
                  individuals.get(ii2).quality().get(finalOI)))
          .toList();
      for (int ii = 1; ii < indexes.size() - 1; ii = ii + 1) {
        int previousIndex = indexes.get(ii - 1);
        int nextIndex = indexes.get(ii + 1);
        double dist = Math.abs(individuals.get(previousIndex).quality().get(finalOI)
            - individuals.get(nextIndex).quality().get(finalOI));
        dists[indexes.get(ii)] = dists[indexes.get(ii)] + dist;
      }
      dists[indexes.get(0)] = dists[indexes.get(0)] + Double.POSITIVE_INFINITY;
      dists[indexes.get(indexes.size() - 1)] = dists[indexes.get(indexes.size() - 1)] + Double.POSITIVE_INFINITY;
    }
    return Arrays.stream(dists).boxed().toList();
  }

  private static <G, S> Collection<RankedIndividual<G, S>> decorate(
      Collection<? extends Individual<G, S, List<Double>>> individuals,
      MultiHomogeneousObjectiveProblem<S, Double> problem) {
    List<? extends Collection<? extends Individual<G, S, List<Double>>>> fronts = PartiallyOrderedCollection.from(
            individuals, partialComparator(problem))
        .fronts();
    return IntStream.range(0, fronts.size())
        .mapToObj(fi -> {
          List<? extends Individual<G, S, List<Double>>> is =
              fronts.get(fi).stream().toList();
          List<Double> distances = distances(is, problem.comparators());
          return IntStream.range(0, is.size())
              .mapToObj(ii -> {
                Individual<G, S, List<Double>> individual = is.get(ii);
                return new RankedIndividual<>(
                    individual.genotype(),
                    individual.solution(),
                    individual.quality(),
                    individual.qualityMappingIteration(),
                    individual.genotypeBirthIteration(),
                    fi,
                    distances.get(ii));
              })
              .toList();
        })
        .flatMap(List::stream)
        .toList();
  }

  @Override
  public POCPopulationState<Individual<G, S, List<Double>>, G, S, List<Double>> init(
      MultiHomogeneousObjectiveProblem<S, Double> problem, RandomGenerator random, ExecutorService executor)
      throws SolverException {
    Collection<? extends Individual<G, S, List<Double>>> individuals =
        map(genotypeFactory.build(populationSize, random), List.of(), null, problem, executor);
    return State.from(
        decorate(individuals, problem).stream()
            .sorted(rankedComparator())
            .toList(),
        partialComparator(problem));
  }

  @Override
  public POCPopulationState<Individual<G, S, List<Double>>, G, S, List<Double>> update(
      MultiHomogeneousObjectiveProblem<S, Double> problem,
      RandomGenerator random,
      ExecutorService executor,
      POCPopulationState<Individual<G, S, List<Double>>, G, S, List<Double>> state)
      throws SolverException {
    State<G, S> listState = (State<G, S>) state;
    // build offspring
    Collection<G> offspringGenotypes = new ArrayList<>();
    Set<G> uniqueOffspringGenotypes = new HashSet<>();
    if (maxUniquenessAttempts > 0) {
      uniqueOffspringGenotypes.addAll(state.pocPopulation().all().stream()
          .map(Individual::genotype)
          .toList());
    }
    int attempts = 0;
    int size = listState.listPopulation().size();
    while (offspringGenotypes.size() < populationSize) {
      GeneticOperator<G> operator = Misc.pickRandomly(operators, random);
      List<G> parentGenotypes = IntStream.range(0, operator.arity())
          .mapToObj(n -> listState
              .listPopulation()
              .get(Math.min(random.nextInt(size), random.nextInt(size)))
              .genotype)
          .toList();
      List<? extends G> childGenotype = operator.apply(parentGenotypes, random);
      if (attempts >= maxUniquenessAttempts
          || childGenotype.stream().noneMatch(uniqueOffspringGenotypes::contains)) {
        attempts = 0;
        offspringGenotypes.addAll(childGenotype);
        uniqueOffspringGenotypes.addAll(childGenotype);
      } else {
        attempts = attempts + 1;
      }
    }
    // map and decorate and trim
    List<RankedIndividual<G, S>> rankedIndividuals =
        decorate(map(offspringGenotypes, state.pocPopulation().all(), state, problem, executor), problem)
            .stream()
            .sorted(rankedComparator())
            .limit(populationSize)
            .toList();
    int nOfNewBirths = offspringGenotypes.size();
    return State.from(
        listState,
        progress(state),
        nOfNewBirths,
        nOfNewBirths + (remap ? populationSize : 0),
        rankedIndividuals,
        partialComparator(problem));
  }

  @Override
  protected Individual<G, S, List<Double>> newIndividual(
      G genotype,
      POCPopulationState<Individual<G, S, List<Double>>, G, S, List<Double>> state,
      MultiHomogeneousObjectiveProblem<S, Double> problem) {
    S solution = solutionMapper.apply(genotype);
    return new RankedIndividual<>(
        genotype,
        solution,
        problem.qualityFunction().apply(solution),
        state == null ? 0 : state.nOfIterations(),
        state == null ? 0 : state.nOfIterations(),
        0,
        0d);
  }

  @Override
  protected Individual<G, S, List<Double>> updateIndividual(
      Individual<G, S, List<Double>> individual,
      POCPopulationState<Individual<G, S, List<Double>>, G, S, List<Double>> state,
      MultiHomogeneousObjectiveProblem<S, Double> problem) {
    return new RankedIndividual<>(
        individual.genotype(),
        individual.solution(),
        problem.qualityFunction().apply(individual.solution()),
        individual.genotypeBirthIteration(),
        state.nOfIterations(),
        0,
        0d);
  }
}
