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
package io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn;

import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import java.util.*;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class NetworkUtils {

  private NetworkUtils() {
  }

  public static List<Integer> compatibleInputPorts(Gate gate, Type type) {
    return IntStream.range(0, gate.inputPorts().size())
        .filter(pi -> gate.inputPorts().get(pi).type().canTakeValuesOf(type))
        .boxed()
        .toList();
  }

  public static List<Integer> compatibleOutputPorts(Gate gate, Type type) {
    return IntStream.range(0, gate.outputTypes().size())
        .filter(pi -> gate.outputTypes().get(pi).canTakeValuesOf(type))
        .boxed()
        .toList();
  }

  public static Comparator<Network> deadComparator() {
    return Comparator.comparingInt(network -> network.deadGates().size());
  }

  public static int freeGateIndex(Network network) {
    return IntStream.range(0, network.gates().size() + 1)
        .filter(gi -> !network.gates().containsKey(gi))
        .min()
        .orElseThrow();
  }

  public static List<Network.Addition> growBothAdditions(Network n, SequencedSet<Gate> gates) {
    List<Network.Addition> additions = new ArrayList<>();
    List<Wire.EndPoint> iEps = new ArrayList<>(n.freeInputEndPoints());
    List<Wire.EndPoint> oEps = new ArrayList<>(n.freeOutputEndPoints());
    int newGateIndex = freeGateIndex(n);
    for (Wire.EndPoint iEp : iEps) {
      for (Wire.EndPoint oEp : oEps) {
        for (Gate gate : gates) {
          for (int ipi = 0; ipi < gate.inputPorts().size(); ipi = ipi + 1) {
            for (int opi = 0; opi < gate.outputTypes().size(); opi = opi + 1) {
              try {
                new Network(
                    List.of(
                        Gate.input(n.concreteOutputType(oEp)),
                        Gate.output(n.concreteInputType(iEp)),
                        gate
                    ),
                    Set.of(
                        Wire.of(0, 0, 2, ipi),
                        Wire.of(2, opi, 1, 0)
                    )
                );
                additions.add(
                    new Network.Addition(
                        Map.of(newGateIndex, gate),
                        Set.of(
                            new Wire(oEp, new Wire.EndPoint(newGateIndex, ipi)),
                            new Wire(new Wire.EndPoint(newGateIndex, opi), iEp)
                        )
                    )
                );
              } catch (NetworkStructureException | TypeException e) {
                // ignore
              }
            }
          }
        }
      }
    }
    return additions;
  }


  public static Network randomHoledNetwork(
      Network n,
      RandomGenerator rnd,
      int targetNOfGates
  ) throws NetworkStructureException, TypeException {
    List<Integer> toRemoveGis = new ArrayList<>();
    List<Integer> availableGis = n.gates()
        .entrySet()
        .stream()
        .filter(e -> !(e.getValue() instanceof Gate.InputGate || e.getValue() instanceof Gate.OutputGate))
        .map(Map.Entry::getKey)
        .toList();
    while (toRemoveGis.size() < targetNOfGates) {
      if (!toRemoveGis.isEmpty()) {
        availableGis = toRemoveGis.stream()
            .flatMap(
                gi -> Stream.concat(
                    n.wiresFrom(gi).stream().map(w -> w.dst().gateIndex()),
                    n.wiresTo(gi).stream().map(w -> w.src().gateIndex())
                )
            )
            .distinct()
            .filter(gi -> !toRemoveGis.contains(gi))
            .filter(
                gi -> !(n.gates().get(gi) instanceof Gate.InputGate) && !(n.gates()
                    .get(gi) instanceof Gate.OutputGate)
            )
            .toList();
      }
      if (availableGis.isEmpty()) {
        break;
      }
      toRemoveGis.add(availableGis.get(rnd.nextInt(availableGis.size())));
    }
    return new Network(
        n.gates()
            .entrySet()
            .stream()
            .filter(e -> !toRemoveGis.contains(e.getKey()))
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    Map.Entry::getValue
                )
            ),
        n.wires()
            .stream()
            .filter(w -> !toRemoveGis.contains(w.src().gateIndex()) && !toRemoveGis.contains(w.dst().gateIndex()))
            .collect(Collectors.toSet())
    );
  }

  public static Network randomSubnetwork(
      Network n,
      RandomGenerator rnd,
      int targetNOfGates
  ) throws NetworkStructureException, TypeException {
    List<Integer> toKeepGis = new ArrayList<>();
    List<Integer> availableGis = n.gates()
        .keySet()
        .stream()
        .filter(
            gi -> !(n.gates().get(gi) instanceof Gate.InputGate) && !(n.gates()
                .get(gi) instanceof Gate.OutputGate)
        )
        .toList();
    while (toKeepGis.size() < targetNOfGates) {
      if (!toKeepGis.isEmpty()) {
        availableGis = toKeepGis.stream()
            .flatMap(
                gi -> Stream.concat(
                    n.wiresFrom(gi).stream().map(w -> w.dst().gateIndex()),
                    n.wiresTo(gi).stream().map(w -> w.src().gateIndex())
                )
            )
            .distinct()
            .filter(gi -> !toKeepGis.contains(gi))
            .filter(
                gi -> !(n.gates().get(gi) instanceof Gate.InputGate || n.gates()
                    .get(gi) instanceof Gate.OutputGate)
            )
            .toList();
      }
      if (availableGis.isEmpty()) {
        break;
      }
      toKeepGis.add(availableGis.get(rnd.nextInt(availableGis.size())));
    }
    return new Network(
        toKeepGis.stream().collect(Collectors.toMap(gi -> gi, gi -> n.gates().get(gi))),
        n.wires()
            .stream()
            .filter(w -> toKeepGis.contains(w.src().gateIndex()) && toKeepGis.contains(w.dst().gateIndex()))
            .collect(Collectors.toSet())
    );
  }

  private static List<Gate> suitableGates(
      SequencedSet<Gate> gates,
      SequencedSet<Type> oTypes,
      SequencedSet<Type> iTypes
  ) {
    List<Gate> oCompatibleGates = gates.stream()
        .filter(
            g -> g.inputPorts()
                .stream()
                .anyMatch(p -> oTypes.stream().anyMatch(t -> p.type().canTakeValuesOf(t)))
        )
        .toList();
    List<Gate> iCompatibleGates = gates.stream()
        .filter(
            g -> g.outputTypes()
                .stream()
                .anyMatch(ot -> iTypes.stream().anyMatch(ot::canTakeValuesOf))
        )
        .toList();
    List<Gate> ioCompatibleGates = new ArrayList<>(iCompatibleGates);
    ioCompatibleGates.retainAll(oCompatibleGates);
    if (!ioCompatibleGates.isEmpty()) {
      return ioCompatibleGates;
    }
    if (!iCompatibleGates.isEmpty()) {
      return iCompatibleGates;
    }
    if (!oCompatibleGates.isEmpty()) {
      return oCompatibleGates;
    }
    return gates.stream().toList();
  }

  public static Network.Addition wire(
      Network n,
      boolean forceDifferentSubnets,
      RandomGenerator rnd
  ) {
    record EnrichedEndPoint(int subnetIndex, int nOfWires, Type type, Wire.EndPoint endPoint) {}
    List<EnrichedEndPoint> oEeps = new ArrayList<>();
    List<EnrichedEndPoint> iEeps = new ArrayList<>();
    List<List<Integer>> subnetsGis = n.disjointSubnetworksGateIndexes();
    for (int si = 0; si < subnetsGis.size(); si = si + 1) {
      for (int gi : subnetsGis.get(si)) {
        for (int pi = 0; pi < n.gates().get(gi).outputTypes().size(); pi = pi + 1) {
          Wire.EndPoint ep = new Wire.EndPoint(gi, pi);
          oEeps.add(
              new EnrichedEndPoint(
                  si,
                  n.wiresFrom(gi, pi).size(),
                  n.concreteOutputType(ep),
                  ep
              )
          );
        }
        for (int pi = 0; pi < n.gates().get(gi).inputPorts().size(); pi = pi + 1) {
          Wire.EndPoint ep = new Wire.EndPoint(gi, pi);
          if (n.wireTo(gi, pi).isEmpty()) {
            iEeps.add(
                new EnrichedEndPoint(
                    si,
                    0,
                    n.concreteInputType(ep),
                    ep
                )
            );
          }
        }
      }
    }
    // shuffle
    Collections.shuffle(oEeps, rnd);
    Collections.shuffle(iEeps, rnd);
    // iterate
    for (EnrichedEndPoint oEep : oEeps) {
      for (EnrichedEndPoint iEep : iEeps) {
        if (iEep.type.canTakeValuesOf(oEep.type)) {
          if (!forceDifferentSubnets || iEep.subnetIndex != oEep.subnetIndex) {
            return new Network.Addition(Map.of(), Set.of(new Wire(oEep.endPoint, iEep.endPoint)));
          }
        }
      }
    }
    return Network.Addition.empty();
  }

  public static Network wireAll(Network network, boolean avoidDeadGates, RandomGenerator rnd) {
    while (true) {
      List<Network.Addition> additions = wires(network);
      Collections.shuffle(additions, rnd);
      if (additions.isEmpty()) {
        return network;
      }
      boolean added = false;
      for (Network.Addition addition : additions) {
        try {
          Network newNetwork = addition.applyTo(network);
          if (!avoidDeadGates || deadComparator().compare(newNetwork, network) < 0) {
            added = true;
            network = newNetwork;
            break;
          }
        } catch (NetworkStructureException | TypeException e) {
          // ignore
        }
      }
      if (!added) {
        break;
      }
    }
    return network;
  }

  public static List<Network.Addition> wires(Network n) {
    List<Network.Addition> additions = new ArrayList<>();
    for (Wire.EndPoint iEp : n.freeInputEndPoints()) {
      Type iType = n.concreteInputType(iEp);
      for (int gi : n.gates().keySet()) {
        for (int pi = 0; pi < n.gates().get(gi).outputTypes().size(); pi = pi + 1) {
          Wire.EndPoint oEp = new Wire.EndPoint(gi, pi);
          Type oType = n.concreteOutputType(oEp);
          if (iType.canTakeValuesOf(oType)) {
            additions.add(new Network.Addition(Map.of(), Set.of(new Wire(oEp, iEp))));
          }
        }
      }
    }
    return additions;
  }

}
