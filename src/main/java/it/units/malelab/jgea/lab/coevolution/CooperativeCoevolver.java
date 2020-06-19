/*
 * Copyright (C) 2020 Eric Medvet <eric.medvet@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package it.units.malelab.jgea.lab.coevolution;

import com.google.common.base.Stopwatch;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.Problem;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.function.BiFunction;
import it.units.malelab.jgea.core.function.Bounded;
import it.units.malelab.jgea.core.function.CachedBoundedNonDeterministicFunction;
import it.units.malelab.jgea.core.function.CachedNonDeterministicFunction;
import it.units.malelab.jgea.core.function.Function;
import it.units.malelab.jgea.core.function.FunctionException;
import it.units.malelab.jgea.core.function.NonDeterministicFunction;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.event.Event;
import it.units.malelab.jgea.core.listener.event.EvolutionEvent;
import it.units.malelab.jgea.core.ranker.Ranker;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * @author Eric Medvet <eric.medvet@gmail.com>
 */
public class CooperativeCoevolver<G1, G2, S1, S2, S, F> implements Evolver<Pair<S1, S2>, S, F> {

  private final Function<Collection<F>, F> fitnessAggregator;
  private final BiFunction<S1, S2, S> composer;
  private final Evolver<G1, S1, F> evolver1;
  private final Evolver<G2, S2, F> evolver2;
  private final Factory<S1> factory1;
  private final Factory<S2> factory2;
  private final Ranker<Individual<Pair<S1, S2>, S, F>> ranker;
  private final int initialPopulationSize;
  private final int cacheSize;

  private static final Logger L = Logger.getLogger(CooperativeCoevolver.class.getName());

  public CooperativeCoevolver(Function<Collection<F>, F> fitnessAggregator, BiFunction<S1, S2, S> composer, Evolver<G1, S1, F> evolver1, Evolver<G2, S2, F> evolver2, Factory<S1> factory1, Factory<S2> factory2, Ranker<Individual<Pair<S1, S2>, S, F>> ranker, int initialPopulationSize, int cacheSize) {
    this.fitnessAggregator = fitnessAggregator;
    this.composer = composer;
    this.evolver1 = evolver1;
    this.evolver2 = evolver2;
    this.factory1 = factory1;
    this.factory2 = factory2;
    this.ranker = ranker;
    this.initialPopulationSize = initialPopulationSize;
    this.cacheSize = cacheSize;
  }

  @Override
  public Collection<S> solve(Problem<S, F> problem, Random random, ExecutorService executor, Listener listener) throws InterruptedException, ExecutionException {
    L.fine("Starting");
    final AtomicInteger fitnessEvaluations = new AtomicInteger();
    final AtomicInteger iterations = new AtomicInteger();
    Stopwatch stopwatch = Stopwatch.createStarted();
    SortedMap<Integer, List<Individual<Pair<S1, S2>, S, F>>> populations = new TreeMap<>();
    SortedMap<Integer, List<S1>> populations1 = new TreeMap<>();
    SortedMap<Integer, List<S2>> populations2 = new TreeMap<>();
    //prepare fitness function
    final NonDeterministicFunction<S, F> fitnessFunction;
    if (cacheSize > 0) {
      if (problem.getFitnessFunction() instanceof Bounded) {
        fitnessFunction = new CachedBoundedNonDeterministicFunction<>(problem.getFitnessFunction(), cacheSize);
      } else {
        fitnessFunction = problem.getFitnessFunction().cached(cacheSize);
      }
    } else {
      fitnessFunction = problem.getFitnessFunction();
    }
    populations1.put(0, factory1.build(initialPopulationSize, random));
    populations2.put(0, factory2.build(initialPopulationSize, random));
    //prepare problems
    final Problem<S1, F> problem1 = () -> createPartialFitnessFunction(
        fitnessFunction,
        composer,
        populations1,
        populations2,
        fitnessEvaluations,
        iterations,
        Function.identity(),
        populations,
        executor,
        random
    );
    final Problem<S2, F> problem2 = () -> createPartialFitnessFunction(
        fitnessFunction,
        (S2 s2, S1 s1, Listener l) -> composer.apply(s1, s2),
        populations2,
        populations1,
        fitnessEvaluations,
        iterations,
        (Pair<S2, S1> p, Listener l) -> Pair.build(p.second(), p.first()),
        populations,
        executor,
        random
    );
    //prepare listeners
    final Listener listener1 = createListener(
        fitnessFunction,
        populations1,
        populations2,
        fitnessEvaluations,
        iterations,
        stopwatch,
        populations,
        listener,
        random
    );
    final Listener listener2 = createListener(
        fitnessFunction,
        populations2,
        populations1,
        fitnessEvaluations,
        iterations,
        stopwatch,
        populations,
        listener,
        random
    );
    //prepare executor services
    final ExecutorService executor1 = Executors.newFixedThreadPool(2);
    final ExecutorService executor2 = Executors.newFixedThreadPool(2);
    //run inner evolvers
    Future<Collection<S1>> future1 = executor1.submit(() -> evolver1.solve(problem1, random, executor1, listener1));
    Future<Collection<S2>> future2 = executor2.submit(() -> evolver2.solve(problem2, random, executor2, listener2));
    //combine solutions
    Collection<S1> solutions1 = future1.get();
    Collection<S2> solutions2 = future2.get();
    L.fine("Ending");
    executor1.shutdown();
    executor2.shutdown();
    Collection<Individual<Pair<S1, S2>, S, F>> bestRank = ranker.rank(populations.get(populations.lastKey()), random).get(0);
    return bestRank.stream()
        .map(Individual::getSolution)
        .collect(Collectors.toList());
  }

  private <ST, SO> Function<ST, F> createPartialFitnessFunction(
      final NonDeterministicFunction<S, F> fitnessFunction,
      final BiFunction<ST, SO, S> composer,
      final SortedMap<Integer, List<ST>> thisPopulations,
      final SortedMap<Integer, List<SO>> otherPopulations,
      final AtomicInteger fitnessEvaluations,
      final AtomicInteger iterations,
      final Function<Pair<ST, SO>, Pair<S1, S2>> pairMapper,
      final SortedMap<Integer, List<Individual<Pair<S1, S2>, S, F>>> composedPopulations,
      final ExecutorService executor,
      final Random random
  ) {
    return (ST st, Listener listener) -> {
      final int thisCurrentCounter = thisPopulations.lastKey();
      while (otherPopulations.lastKey() < thisCurrentCounter) {
        synchronized (otherPopulations) {
          try {
            otherPopulations.wait(100); //TODO maybe put a constant
          } catch (InterruptedException ex) {
            //ignore
          }
        }
      }
      List<Callable<F>> callables;
      synchronized (otherPopulations) {
        try {
          callables = otherPopulations.get(thisCurrentCounter).stream().map(so -> (Callable<F>) () -> {
            S solution = composer.apply(st, so);
            fitnessEvaluations.incrementAndGet();
            F fitness = fitnessFunction.apply(solution, random);
            Individual<Pair<S1, S2>, S, F> individual = new Individual<>(
                pairMapper.apply(Pair.build(st, so)),
                solution,
                fitness,
                iterations.get(),
                Collections.EMPTY_LIST,
                Collections.EMPTY_MAP
            );
            List<Individual<Pair<S1, S2>, S, F>> composedPopulation;
            synchronized (composedPopulations) {
              composedPopulation = composedPopulations.get(thisCurrentCounter);
              if (composedPopulation == null) {
                composedPopulation = new ArrayList<>();
                composedPopulations.put(thisCurrentCounter, composedPopulation);
              }
              composedPopulation.add(individual);
            }
            return fitness;
          }).collect(Collectors.toList());
        } catch (Throwable t) {
          L.severe(String.format("gt=%2d, go=%2d, gc=%2d\tf() of %s", thisPopulations.lastKey(), otherPopulations.lastKey(), composedPopulations.lastKey(), st));
          throw new FunctionException(t);
        }
      }
      List<F> fs;
      try {
        fs = Misc.getAll(executor.invokeAll(callables));
      } catch (InterruptedException | ExecutionException ex) {
        throw new FunctionException(ex);
      }
      return fitnessAggregator.apply(fs);
    };
  }

  private <ST, SO> Listener createListener(
      final NonDeterministicFunction<S, F> fitnessFunction,
      final SortedMap<Integer, List<ST>> thisPopulations,
      final SortedMap<Integer, List<SO>> otherPopulations,
      final AtomicInteger fitnessEvaluations,
      final AtomicInteger iterations,
      final Stopwatch stopwatch,
      final SortedMap<Integer, List<Individual<Pair<S1, S2>, S, F>>> composedPopulations,
      final Listener listener,
      final Random random
  ) {
    return (Event event) -> {
      if (event instanceof EvolutionEvent) {
        List<ST> population = new ArrayList<>();
        for (Collection<? extends Individual> rank : ((EvolutionEvent) event).getRankedPopulation()) {
          rank.stream().map(i -> (ST) i.getSolution()).forEach(st -> population.add(st));
        }
        synchronized (thisPopulations) {
          thisPopulations.put(((EvolutionEvent) event).getIteration(), population);
          thisPopulations.notifyAll();
          if (thisPopulations.lastKey().equals(otherPopulations.lastKey())) {
            iterations.set(thisPopulations.lastKey());
            EvolutionEvent composedEvent = new EvolutionEvent(
                iterations.get(),
                fitnessEvaluations.get(),
                (fitnessFunction instanceof CachedNonDeterministicFunction) ? ((CachedNonDeterministicFunction) fitnessFunction).getActualCount() : fitnessEvaluations.get(),
                (List) ranker.rank(composedPopulations.get(iterations.get() - 1), random),
                stopwatch.elapsed(TimeUnit.MILLISECONDS)
            );
            listener.listen(composedEvent);
          }
        }
      }
    };
  }

}