/*-
 * ========================LICENSE_START=================================
 * jgea-core
 * %%
 * Copyright (C) 2018 - 2025 Eric Medvet
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
package io.github.ericmedvet.jgea.core.solver.bi;

import io.github.ericmedvet.jgea.core.Factory;
import io.github.ericmedvet.jgea.core.operator.GeneticOperator;
import io.github.ericmedvet.jgea.core.order.DAGPartiallyOrderedCollection;
import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.order.PartiallyOrderedCollection;
import io.github.ericmedvet.jgea.core.problem.QualityBasedBiProblem;
import io.github.ericmedvet.jgea.core.selector.Selector;
import io.github.ericmedvet.jgea.core.solver.Individual;
import io.github.ericmedvet.jgea.core.solver.POCPopulationState;
import io.github.ericmedvet.jgea.core.solver.SolverException;
import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.random.RandomGenerator;

public class StandardBiEvolver<G, S, Q, O> extends AbstractBiEvolver<POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>>, QualityBasedBiProblem<S, O, Q>, Individual<G, S, Q>, G, S, Q, O> {

  protected final Map<GeneticOperator<G>, Double> operators;
  protected final Selector<? super Individual<G, S, Q>> parentSelector;
  protected final Selector<? super Individual<G, S, Q>> unsurvivalSelector;
  protected final int populationSize;
  protected final int offspringSize;
  protected final boolean overlapping;
  protected final int maxUniquenessAttempts;

  public StandardBiEvolver(
      Function<? super G, ? extends S> solutionMapper,
      Factory<? extends G> genotypeFactory,
      int populationSize,
      Predicate<? super POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>>> stopCondition,
      Map<GeneticOperator<G>, Double> operators,
      Selector<? super Individual<G, S, Q>> parentSelector,
      Selector<? super Individual<G, S, Q>> unsurvivalSelector,
      int offspringSize,
      boolean overlapping,
      int maxUniquenessAttempts,
      BinaryOperator<Q> fitnessReducer,
      List<PartialComparator<? super Individual<G, S, Q>>> additionalIndividualComparators
  ) {
    super(solutionMapper, genotypeFactory, stopCondition, false, fitnessReducer, additionalIndividualComparators);
    this.operators = operators;
    this.parentSelector = parentSelector;
    this.unsurvivalSelector = unsurvivalSelector;
    this.populationSize = populationSize;
    this.offspringSize = offspringSize;
    this.overlapping = overlapping;
    this.maxUniquenessAttempts = maxUniquenessAttempts;
  }

  protected Collection<ChildGenotype<G>> buildOffspringToMapGenotypes(
      POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>> state,
      RandomGenerator random
  ) {
    AtomicLong counter = new AtomicLong(state.nOfBirths());
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
    while (offspringChildGenotypes.size() < offspringSize) {
      GeneticOperator<G> operator = Misc.pickRandomly(operators, random);
      List<Individual<G, S, Q>> parents = new ArrayList<>(operator.arity());
      for (int j = 0; j < operator.arity(); j++) {
        Individual<G, S, Q> parent = parentSelector.select(state.pocPopulation(), random);
        parents.add(parent);
      }
      List<? extends G> childGenotypes = operator.apply(parents.stream().map(Individual::genotype).toList(), random);
      if (attempts >= maxUniquenessAttempts || childGenotypes.stream().noneMatch(uniqueOffspringGenotypes::contains)) {
        attempts = 0;
        List<Long> parentIds = parents.stream().map(Individual::id).toList();
        childGenotypes.stream()
            .map(g -> new ChildGenotype<G>(counter.getAndIncrement(), g, parentIds))
            .forEach(offspringChildGenotypes::add);
        uniqueOffspringGenotypes.addAll(childGenotypes);
      } else {
        attempts = attempts + 1;
      }
    }
    return offspringChildGenotypes;
  }

  protected POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>> init(
      QualityBasedBiProblem<S, O, Q> problem
  ) {
    return POCPopulationState.empty(problem, stopCondition());
  }

  @Override
  public POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>> init(
      QualityBasedBiProblem<S, O, Q> problem,
      RandomGenerator random,
      ExecutorService executor
  ) throws SolverException {
    AtomicLong counter = new AtomicLong(0);
    List<? extends G> genotypes = genotypeFactory.build(populationSize, random);
    List<? extends G> firstHalfGenotypes;
    int half = populationSize / 2;
    if (populationSize % 2 == 0) {
      firstHalfGenotypes = genotypes.subList(0, half);
    } else {
      firstHalfGenotypes = genotypes.subList(0, half + 1);
    }
    List<? extends G> secondHalfGenotypes = genotypes.subList(half, populationSize);
    List<Future<List<Individual<G, S, Q>>>> futures = new ArrayList<>();
    for (int i = 0; i < firstHalfGenotypes.size(); i++) {
      int index = i;
      Future<List<Individual<G, S, Q>>> future = executor.submit(() -> {
        G firstGenotype = firstHalfGenotypes.get(index);
        G secondGenotype = secondHalfGenotypes.get(index);
        S firstSolution = solutionMapper.apply(firstGenotype);
        S secondSolution = solutionMapper.apply(secondGenotype);
        O outcome = problem.outcomeFunction().apply(firstSolution, secondSolution);
        Q firstQuality = problem.firstQualityFunction().apply(outcome);
        Q secondQuality = problem.secondQualityFunction().apply(outcome);
        Individual<G, S, Q> firstIndividual = Individual.of(
            counter.getAndIncrement(),
            firstGenotype,
            firstSolution,
            firstQuality,
            0,
            0,
            List.of()
        );
        Individual<G, S, Q> secondIndividual = Individual.of(
            counter.getAndIncrement(),
            secondGenotype,
            secondSolution,
            secondQuality,
            0,
            0,
            List.of()
        );
        return List.of(firstIndividual, secondIndividual);
      });
      futures.add(future);
    }
    Collection<Individual<G, S, Q>> individuals = futures.stream()
        .map(future -> {
          try {
            return future.get();
          } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
          }
        })
        .flatMap(Collection::stream)
        .toList();
    POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>> state = POCPopulationState.empty(
        problem,
        stopCondition()
    );
    return state.updatedWithIteration(
        populationSize,
        futures.size(),
        PartiallyOrderedCollection.from(individuals, partialComparator(problem))
    );
  }

  @Override
  public POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>> update(
      RandomGenerator random,
      ExecutorService executor,
      POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>> state
  ) throws SolverException {
    // create offspring genotypes with empty solution and quality
    List<ChildGenotype<G>> offspringChildGenotypes = buildOffspringToMapGenotypes(state, random).stream().toList();
    int nOfNewBirths = offspringChildGenotypes.size();
    List<Individual<G, S, Q>> individuals = new ArrayList<>(
        offspringChildGenotypes.stream()
            .map(g -> Individual.<G, S, Q>from(g, solutionMapper, s -> null, state.nOfIterations()))
            .toList()
    );
    individuals.addAll(state.pocPopulation().all().stream().toList());
    // randomly "pair" individuals
    Collections.shuffle(individuals, random);
    List<Individual<G, S, Q>> firstHalfIndividuals;
    int half = individuals.size() / 2;
    if (individuals.size() % 2 == 0) {
      firstHalfIndividuals = individuals.subList(0, half);
    } else {
      firstHalfIndividuals = individuals.subList(0, half + 1);
    }
    List<Individual<G, S, Q>> secondHalfIndividuals = individuals.subList(half, individuals.size());
    // prepare futures
    List<Future<List<Individual<G, S, Q>>>> futures = new ArrayList<>();
    for (int i = 0; i < firstHalfIndividuals.size(); i++) {
      int index = i;
      Future<List<Individual<G, S, Q>>> future = executor.submit(() -> {
        Individual<G, S, Q> firstIndividual = firstHalfIndividuals.get(index);
        Individual<G, S, Q> secondIndividual = secondHalfIndividuals.get(index);
        O outcome = state.problem()
            .outcomeFunction()
            .apply(firstIndividual.solution(), secondIndividual.solution());
        Q newFirstQuality = state.problem().firstQualityFunction().apply(outcome);
        Q newSecondQuality = state.problem().secondQualityFunction().apply(outcome);
        firstIndividual = Objects.isNull(firstIndividual.quality()) ? firstIndividual.updateQuality(
            newFirstQuality,
            state.nOfIterations()
        ) : firstIndividual.updateQuality(
            fitnessReducer.apply(firstIndividual.quality(), newFirstQuality),
            state.nOfIterations()
        );
        secondIndividual = Objects.isNull(secondIndividual.quality()) ? secondIndividual.updateQuality(
            newSecondQuality,
            state.nOfIterations()
        ) : secondIndividual.updateQuality(
            fitnessReducer.apply(secondIndividual.quality(), newSecondQuality),
            state.nOfIterations()
        );
        // If populationSize is odd, the last element of firstHalfGenotypes is the same as the first element of
        // secondHalfGenotypes
        if (individuals.size() % 2 != 0 && index != 0) {
          return List.of(firstIndividual);
        }
        return List.of(firstIndividual, secondIndividual);
      });
      futures.add(future);
    }
    // extract future results
    Collection<Individual<G, S, Q>> newPopulation = futures.stream()
        .map(listFuture -> {
          try {
            return listFuture.get();
          } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
          }
        })
        .flatMap(Collection::stream)
        .toList();
    newPopulation = trimPopulation(newPopulation, state, random);
    return state.updatedWithIteration(
        nOfNewBirths,
        futures.size(),
        PartiallyOrderedCollection.from(newPopulation, partialComparator(state.problem()))
    );
  }

  protected Collection<Individual<G, S, Q>> trimPopulation(
      Collection<Individual<G, S, Q>> population,
      POCPopulationState<Individual<G, S, Q>, G, S, Q, QualityBasedBiProblem<S, O, Q>> state,
      RandomGenerator random
  ) {
    PartiallyOrderedCollection<Individual<G, S, Q>> orderedPopulation = new DAGPartiallyOrderedCollection<>(
        population,
        partialComparator(state.problem())
    );
    while (orderedPopulation.size() > populationSize) {
      Individual<G, S, Q> toRemoveIndividual = unsurvivalSelector.select(orderedPopulation, random);
      orderedPopulation.remove(toRemoveIndividual);
    }
    return orderedPopulation.all();
  }
}
