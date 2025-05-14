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
package io.github.ericmedvet.jgea.core.solver.bi.mapelites;

import io.github.ericmedvet.jgea.core.Factory;
import io.github.ericmedvet.jgea.core.operator.Mutation;
import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.problem.QualityBasedBiProblem;
import io.github.ericmedvet.jgea.core.solver.Individual;
import io.github.ericmedvet.jgea.core.solver.SolverException;
import io.github.ericmedvet.jgea.core.solver.bi.AbstractBiEvolver;
import io.github.ericmedvet.jgea.core.solver.mapelites.Archive;
import io.github.ericmedvet.jgea.core.solver.mapelites.MEIndividual;
import io.github.ericmedvet.jgea.core.solver.mapelites.MEPopulationState;
import io.github.ericmedvet.jgea.core.solver.mapelites.MapElites;
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
import java.util.stream.IntStream;

public class MapElitesBiEvolver<G, S, Q, O> extends AbstractBiEvolver<MEPopulationState<G, S, Q, QualityBasedBiProblem<S, O, Q>>, QualityBasedBiProblem<S, O, Q>, MEIndividual<G, S, Q>, G, S, Q, O> {

  protected final int populationSize;
  private final Mutation<G> mutation;
  private final List<MapElites.Descriptor<G, S, Q>> descriptors;
  private final boolean emptyArchive;

  public MapElitesBiEvolver(
      Function<? super G, ? extends S> solutionMapper,
      Factory<? extends G> genotypeFactory,
      Predicate<? super MEPopulationState<G, S, Q, QualityBasedBiProblem<S, O, Q>>> stopCondition,
      Mutation<G> mutation,
      int populationSize,
      List<MapElites.Descriptor<G, S, Q>> descriptors,
      BinaryOperator<Q> fitnessReducer,
      boolean emptyArchive,
      List<PartialComparator<? super MEIndividual<G, S, Q>>> additionalIndividualComparators
  ) {
    super(solutionMapper, genotypeFactory, stopCondition, false, fitnessReducer, additionalIndividualComparators);
    this.populationSize = populationSize;
    this.mutation = mutation;
    this.descriptors = descriptors;
    this.emptyArchive = emptyArchive;
  }

  @Override
  public MEPopulationState<G, S, Q, QualityBasedBiProblem<S, O, Q>> init(
      QualityBasedBiProblem<S, O, Q> problem,
      RandomGenerator random,
      ExecutorService executor
  ) throws SolverException {
    // create new genotypes and split in two list of opponents
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
    // build futures
    List<Future<List<MEIndividual<G, S, Q>>>> futures = new ArrayList<>();
    for (int i = 0; i < firstHalfGenotypes.size(); i++) {
      int index = i;
      Future<List<MEIndividual<G, S, Q>>> future = executor.submit(() -> {
        G firstGenotype = firstHalfGenotypes.get(index);
        G secondGenotype = secondHalfGenotypes.get(index);
        S firstSolution = solutionMapper.apply(firstGenotype);
        S secondSolution = solutionMapper.apply(secondGenotype);
        O outcome = problem.outcomeFunction().apply(firstSolution, secondSolution);
        Q firstQuality = problem.firstQualityFunction().apply(outcome);
        Q secondQuality = problem.secondQualityFunction().apply(outcome);
        MEIndividual<G, S, Q> firstIndividual = MEIndividual.from(
            Individual.of(
                counter.getAndIncrement(),
                firstGenotype,
                firstSolution,
                firstQuality,
                0,
                0,
                List.of()
            ),
            descriptors
        );
        MEIndividual<G, S, Q> secondIndividual = MEIndividual.from(
            Individual.of(
                counter.getAndIncrement(),
                secondGenotype,
                secondSolution,
                secondQuality,
                0,
                0,
                List.of()
            ),
            descriptors
        );
        return List.of(firstIndividual, secondIndividual);
      });
      futures.add(future);
    }
    // extract future results
    Collection<MEIndividual<G, S, Q>> individuals = futures.stream()
        .map(listFuture -> {
          try {
            return listFuture.get();
          } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
          }
        })
        .flatMap(Collection::stream)
        .toList();
    MEPopulationState<G, S, Q, QualityBasedBiProblem<S, O, Q>> newState = MEPopulationState.empty(
        problem,
        stopCondition(),
        descriptors
    );
    return newState.updatedWithIteration(
        populationSize,
        futures.size(),
        newState.archive().updated(individuals, MEIndividual::bins, partialComparator(problem))
    );
  }

  @Override
  public MEPopulationState<G, S, Q, QualityBasedBiProblem<S, O, Q>> update(
      RandomGenerator random,
      ExecutorService executor,
      MEPopulationState<G, S, Q, QualityBasedBiProblem<S, O, Q>> state
  ) throws SolverException {
    // build offspring with empty quality
    List<MEIndividual<G, S, Q>> individuals = new ArrayList<>(state.archive().asMap().values().stream().toList());
    AtomicLong counter = new AtomicLong(state.nOfBirths());
    List<ChildGenotype<G>> newChildGenotypes = IntStream.range(0, populationSize)
        .mapToObj(j -> Misc.pickRandomly(individuals, random))
        .map(
            p -> new ChildGenotype<>(
                counter.getAndIncrement(),
                mutation.mutate(p.genotype(), random),
                List.of(p.id())
            )
        )
        .toList();
    individuals.addAll(
        newChildGenotypes.stream()
            .map(
                g -> MEIndividual.from(
                    Individual.from(g, solutionMapper, s -> null, state.nOfQualityEvaluations()),
                    descriptors
                )
            )
            .toList()
    );
    // randomly "pair" individuals
    Collections.shuffle(individuals, random);
    List<MEIndividual<G, S, Q>> firstHalfIndividuals;
    int half = individuals.size() / 2;
    if (individuals.size() % 2 == 0) {
      firstHalfIndividuals = individuals.subList(0, half);
    } else {
      firstHalfIndividuals = individuals.subList(0, half + 1);
    }
    List<MEIndividual<G, S, Q>> secondHalfIndividuals = individuals.subList(half, individuals.size());
    // build futures
    List<Future<List<MEIndividual<G, S, Q>>>> futures = new ArrayList<>();
    for (int i = 0; i < firstHalfIndividuals.size(); i++) {
      int index = i;
      Future<List<MEIndividual<G, S, Q>>> future = executor.submit(() -> {
        MEIndividual<G, S, Q> firstIndividual = firstHalfIndividuals.get(index);
        MEIndividual<G, S, Q> secondIndividual = secondHalfIndividuals.get(index);
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
    // Collect futures
    Collection<MEIndividual<G, S, Q>> newAndOldPopulation = futures.stream()
        .map(listFuture -> {
          try {
            return listFuture.get();
          } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
          }
        })
        .flatMap(Collection::stream)
        .toList();
    PartialComparator<? super Individual<?, ?, ?>> updaterComparator = (
        newI,
        existingI
    ) -> newI == existingI ? PartialComparator.PartialComparatorOutcome.BEFORE : PartialComparator.PartialComparatorOutcome.AFTER;
    Archive<MEIndividual<G, S, Q>> archive;
    if (emptyArchive) {
      archive = new Archive<>(state.archive().binUpperBounds());
    } else {
      archive = state.archive().updated(newAndOldPopulation, MEIndividual::bins, updaterComparator);
    }
    archive = archive.updated(newAndOldPopulation, MEIndividual::bins, partialComparator(state.problem()));
    return state.updatedWithIteration(populationSize, futures.size(), archive);
  }
}
