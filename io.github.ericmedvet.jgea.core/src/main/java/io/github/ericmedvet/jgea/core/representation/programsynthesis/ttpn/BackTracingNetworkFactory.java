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
package io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn;

import io.github.ericmedvet.jgea.core.IndependentFactory;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Generic;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import io.github.ericmedvet.jgea.core.util.LinkedHashMultiset;
import io.github.ericmedvet.jgea.core.util.Misc;
import io.github.ericmedvet.jgea.core.util.Multiset;
import io.github.ericmedvet.jnb.datastructure.Pair;
import java.util.*;
import java.util.function.Function;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class BackTracingNetworkFactory implements IndependentFactory<Network> {
  private final List<Type> inputTypes;
  private final List<Type> outputTypes;
  private final SequencedSet<Gate> gates;
  private final int maxNOfGates;
  private final int maxNOfAttempts;

  public BackTracingNetworkFactory(
      List<Type> inputTypes,
      List<Type> outputTypes,
      SequencedSet<Gate> gates,
      int maxNOfGates,
      int maxNOfAttempts
  ) {
    this.inputTypes = inputTypes;
    this.outputTypes = outputTypes;
    this.gates = gates;
    this.maxNOfGates = maxNOfGates;
    this.maxNOfAttempts = maxNOfAttempts;
  }

  private record AugmentedType(Type type, boolean atLeastZero) {}

  private record Step(Network network, Queue<SequencedSet<Network.Addition>> additionSets) {}

  private record WeightedAdditionProvider(
      Function<Network, List<SequencedSet<Network.Addition>>> provider, double weight
  ) {}

  private static Step computeStep(
      Network network,
      List<WeightedAdditionProvider> providers,
      RandomGenerator random
  ) {
    record WeightedAdditionSet(SequencedSet<Network.Addition> additions, double weight) {}
    List<WeightedAdditionSet> weightedAdditionSets = providers.stream()
        .flatMap(
            wProvider -> wProvider.provider.apply(network)
                .stream()
                .map(addition -> new WeightedAdditionSet(addition, wProvider.weight))
        )
        .toList();
    int[] indexes = IntStream.range(0, weightedAdditionSets.size()).toArray();
    // random shuffle
    IntStream.range(0, indexes.length)
        .forEach(
            i -> swap(
                indexes,
                random.nextInt(indexes.length),
                random.nextInt(indexes.length)
            )
        );
    // weighted shuffle
    IntStream.range(0, indexes.length).forEach(i -> {
      int i1 = random.nextInt(indexes.length - 1);
      int i2 = random.nextInt(i1, indexes.length - 1);
      double w1 = weightedAdditionSets.get(i1).weight;
      double w2 = weightedAdditionSets.get(i2).weight;
      double x = random.nextDouble(w1 + w2);
      if (x > w1) {
        swap(indexes, i1, i2);
      }
    });
    return new Step(
        network,
        Arrays.stream(indexes)
            .mapToObj(i -> weightedAdditionSets.get(i).additions)
            .collect(Collectors.toCollection(LinkedList::new))
    );
  }

  private static List<SequencedSet<Network.Addition>> inTheMiddleGateAdditions(
      Network network,
      SequencedSet<Gate> gates
  ) {
    int newGateIndex = NetworkUtils.freeGateIndex(network);
    List<SequencedSet<Network.Addition>> additionSets = new ArrayList<>();
    for (Wire.EndPoint dstEp : network.freeInputEndPoints()) {
      Type dstType = network.concreteInputType(dstEp);
      for (Map.Entry<Integer, Gate> srcGateEntry : network.gates().entrySet()) {
        for (int srcOpi = 0; srcOpi < srcGateEntry.getValue().outputTypes().size(); srcOpi = srcOpi + 1) {
          Type srcType = network.concreteOutputType(new Wire.EndPoint(srcGateEntry.getKey(), srcOpi));
          Map<Pair<Multiset<AugmentedType>, Multiset<Type>>, SequencedSet<Network.Addition>> groupedAdditions = new LinkedHashMap<>();
          for (Gate newGate : gates) {
            for (int ipi = 0; ipi < newGate.inputPorts().size(); ipi = ipi + 1) {
              for (int opi = 0; opi < newGate.outputTypes().size(); opi = opi + 1) {
                Type ngiType = newGate.inputPorts().get(ipi).type();
                if (ngiType.canTakeValuesOf(srcType)) {
                  Map<Generic, Type> generics = safelyResolve(srcType, ngiType);
                  Type ngoType = newGate.outputTypes().get(opi).concrete(generics);
                  if (dstType.canTakeValuesOf(ngoType)) {
                    Pair<Multiset<AugmentedType>, Multiset<Type>> key = key(newGate, ipi, opi, generics);
                    groupedAdditions
                        .computeIfAbsent(key, k -> new LinkedHashSet<>())
                        .add(
                            new Network.Addition(
                                Map.of(newGateIndex, newGate),
                                Set.of(
                                    Wire.of(srcGateEntry.getKey(), srcOpi, newGateIndex, ipi),
                                    Wire.of(newGateIndex, opi, dstEp.gateIndex(), dstEp.portIndex())
                                )
                            )
                        );
                  }
                }
              }
            }
          }
          additionSets.addAll(groupedAdditions.values());
        }
      }
    }
    return additionSets;
  }

  private static Pair<Multiset<AugmentedType>, Multiset<Type>> key(
      Gate newGate,
      int ipi,
      int opi,
      Map<Generic, Type> generics
  ) {
    Multiset<AugmentedType> unusedInputs = IntStream.range(0, newGate.inputPorts().size())
        .filter(i -> i != ipi)
        .mapToObj(
            i -> new AugmentedType(
                newGate.inputPorts().get(i).type().concrete(generics),
                newGate.inputPorts().get(i).n() == 0
            )
        )
        .collect(Collectors.toCollection(LinkedHashMultiset::new));
    Multiset<Type> unusedOutputs = IntStream.range(0, newGate.outputTypes().size())
        .filter(i -> i != opi)
        .mapToObj(i -> newGate.outputTypes().get(i).concrete(generics))
        .collect(Collectors.toCollection(LinkedHashMultiset::new));
    return new Pair<>(unusedInputs, unusedOutputs);
  }

  private static List<SequencedSet<Network.Addition>> onInputGateAdditions(Network network, SequencedSet<Gate> gates) {
    int newGateIndex = NetworkUtils.freeGateIndex(network);
    List<SequencedSet<Network.Addition>> additionSets = new ArrayList<>();
    for (Wire.EndPoint dstEp : network.freeInputEndPoints()) {
      Type dstType = network.concreteInputType(dstEp);
      Map<Pair<Multiset<AugmentedType>, Multiset<Type>>, SequencedSet<Network.Addition>> groupedAdditions = new LinkedHashMap<>();
      for (Gate newGate : gates) {
        for (int gpi = 0; gpi < newGate.outputTypes().size(); gpi = gpi + 1) {
          Type ngoType = newGate.outputTypes().get(gpi);
          if (dstType.canTakeValuesOf(ngoType)) {
            Pair<Multiset<AugmentedType>, Multiset<Type>> key = key(newGate, -1, gpi, safelyResolve(dstType, ngoType));
            groupedAdditions
                .computeIfAbsent(key, k -> new LinkedHashSet<>())
                .add(
                    new Network.Addition(
                        Map.of(newGateIndex, newGate),
                        Set.of(Wire.of(newGateIndex, gpi, dstEp.gateIndex(), dstEp.portIndex()))
                    )
                );
          }
        }
      }
      additionSets.addAll(groupedAdditions.values());
    }
    return additionSets;
  }

  private static List<SequencedSet<Network.Addition>> onOutputGateAdditions(Network network, SequencedSet<Gate> gates) {
    int newGateIndex = NetworkUtils.freeGateIndex(network);
    List<SequencedSet<Network.Addition>> additionSets = new ArrayList<>();
    for (Map.Entry<Integer, Gate> gateEntry : network.gates().entrySet()) {
      for (int pi = 0; pi < gateEntry.getValue().outputTypes().size(); pi = pi + 1) {
        Type srcType = network.concreteOutputType(new Wire.EndPoint(gateEntry.getKey(), pi));
        Map<Pair<Multiset<AugmentedType>, Multiset<Type>>, SequencedSet<Network.Addition>> groupedAdditions = new LinkedHashMap<>();
        for (Gate newGate : gates) {
          for (int gpi = 0; gpi < newGate.inputPorts().size(); gpi = gpi + 1) {
            Type ngiType = newGate.inputPorts().get(gpi).type();
            if (ngiType.canTakeValuesOf(srcType)) {
              Pair<Multiset<AugmentedType>, Multiset<Type>> key = key(
                  newGate,
                  gpi,
                  -1,
                  safelyResolve(srcType, ngiType)
              );
              groupedAdditions
                  .computeIfAbsent(key, k -> new LinkedHashSet<>())
                  .add(
                      new Network.Addition(
                          Map.of(newGateIndex, newGate),
                          Set.of(Wire.of(gateEntry.getKey(), pi, newGateIndex, gpi))
                      )
                  );
            }
          }
        }
        additionSets.addAll(groupedAdditions.values());
      }
    }
    return additionSets;
  }

  @SafeVarargs
  private static <T> SequencedSet<T> sSet(T... ts) {
    return Arrays.stream(ts).collect(Collectors.toCollection(LinkedHashSet::new));
  }

  public static Map<Generic, Type> safelyResolve(Type concrete, Type type) {
    try {
      return type.resolveGenerics(concrete);
    } catch (TypeException e) {
      return Map.of();
    }
  }

  private static void swap(int[] array, int i, int j) {
    int v = array[i];
    array[i] = array[j];
    array[j] = v;
  }

  private static List<SequencedSet<Network.Addition>> wireAdditions(Network network) {
    List<SequencedSet<Network.Addition>> additionSets = new ArrayList<>();
    for (Wire.EndPoint iEp : network.freeInputEndPoints()) {
      Type iType = network.concreteInputType(iEp);
      for (Map.Entry<Integer, Gate> gateEntry : network.gates().entrySet()) {
        for (int pi = 0; pi < gateEntry.getValue().outputTypes().size(); pi = pi + 1) {
          if (iType.canTakeValuesOf(network.concreteOutputType(new Wire.EndPoint(gateEntry.getKey(), pi)))) {
            additionSets.add(
                sSet(
                    new Network.Addition(
                        Map.of(),
                        Set.of(new Wire(new Wire.EndPoint(gateEntry.getKey(), pi), iEp))
                    )
                )
            );
          }
        }
      }
    }
    return additionSets;
  }

  @Override
  public Network build(RandomGenerator random) {
    // build initial empty network
    Network network;
    try {
      network = new Network(
          Stream.concat(
              inputTypes.stream().map(type -> (Gate) Gate.input(type)),
              outputTypes.stream().map(Gate::output)
          ).toList(),
          Set.of()
      );
    } catch (NetworkStructureException | TypeException e) {
      throw new IllegalArgumentException("Cannot init network", e);
    }
    // do greedy building
    List<WeightedAdditionProvider> additionProviders = List.of(
        new WeightedAdditionProvider(BackTracingNetworkFactory::wireAdditions, 10d),
        new WeightedAdditionProvider(n -> onInputGateAdditions(n, gates), 1d),
        new WeightedAdditionProvider(n -> onOutputGateAdditions(n, gates), 1d),
        new WeightedAdditionProvider(n -> inTheMiddleGateAdditions(n, gates), 5d)
    );
    List<Step> steps = new LinkedList<>();
    steps.addLast(computeStep(network, additionProviders, random));
    int nOfAttempts = 0;
    while (nOfAttempts < maxNOfAttempts) {
      Step step = steps.getLast();
      if (step.additionSets.isEmpty()) {
        steps.removeLast();
        if (steps.isEmpty()) {
          break;
        }
      } else {
        SequencedSet<Network.Addition> additionSet = step.additionSets.remove();
        try {
          Network newNetwork = Misc.pickRandomly(additionSet, random).applyTo(step.network);
          if (newNetwork.deadOrIUnwiredOutputGates().isEmpty()) {
            return newNetwork;
          }
          if (newNetwork.gates().size() <= maxNOfGates) {
            steps.addLast(computeStep(newNetwork, additionProviders, random));
          }
        } catch (NetworkStructureException | TypeException e) {
          // ignore (should happen rarely...)
        }
        nOfAttempts = nOfAttempts + 1;
      }
    }
    return steps.isEmpty() ? network : steps.getLast().network();
  }

}
