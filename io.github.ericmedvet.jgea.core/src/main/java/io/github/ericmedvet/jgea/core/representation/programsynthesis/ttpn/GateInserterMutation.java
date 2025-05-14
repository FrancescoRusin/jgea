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

import io.github.ericmedvet.jgea.core.operator.Mutation;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Generic;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.*;
import java.util.random.RandomGenerator;

public class GateInserterMutation implements Mutation<Network> {
  private final SequencedSet<Gate> gates;
  private final int maxNOfGates;
  private final int maxNOfAttempts;
  private final boolean avoidDeadGates;

  public GateInserterMutation(
      SequencedSet<Gate> gates,
      int maxNOfGates,
      int maxNOfAttempts,
      boolean avoidDeadGates
  ) {
    this.gates = gates;
    this.maxNOfGates = maxNOfGates;
    this.maxNOfAttempts = maxNOfAttempts;
    this.avoidDeadGates = avoidDeadGates;
  }

  @Override
  public Network mutate(Network n, RandomGenerator rnd) {
    if (n.gates().size() >= maxNOfGates || n.wires().isEmpty()) {
      return n;
    }
    List<Gate> shuffledGates = new ArrayList<>(gates);
    Collections.shuffle(shuffledGates, rnd);
    int newGateIndex = NetworkUtils.freeGateIndex(n);
    Map<Integer, Gate> newGates = new HashMap<>(n.gates());
    int nOfAttempts = 0;
    while (nOfAttempts < maxNOfAttempts) {
      // cut one wire
      Wire toRemoveWire = Misc.pickRandomly(n.wires(), rnd);
      Set<Wire> newWires = new HashSet<>(n.wires());
      newWires.remove(toRemoveWire);
      Type srcType = n.concreteOutputType(toRemoveWire.src());
      Type dstType = n.concreteInputType(toRemoveWire.dst());
      for (Gate newGate : shuffledGates) {
        newGates.put(newGateIndex, newGate);
        for (int ipi = 0; ipi < newGate.inputPorts().size(); ipi = ipi + 1) {
          for (int opi = 0; opi < newGate.outputTypes().size(); opi = opi + 1) {
            Type ngiType = newGate.inputPorts().get(ipi).type();
            if (ngiType.canTakeValuesOf(srcType)) {
              Map<Generic, Type> generics = BackTracingNetworkFactory.safelyResolve(srcType, ngiType);
              Type ngoType = newGate.outputTypes().get(opi).concrete(generics);
              if (dstType.canTakeValuesOf(ngoType)) {
                Wire newWireBefore = new Wire(toRemoveWire.src(), new Wire.EndPoint(newGateIndex, ipi));
                Wire newWireAfter = new Wire(new Wire.EndPoint(newGateIndex, opi), toRemoveWire.dst());
                newWires.add(newWireBefore);
                newWires.add(newWireAfter);
                try {
                  Network newNetwork = new Network(newGates, newWires);
                  while (true) {
                    Network.Addition addition = NetworkUtils.wire(newNetwork, false, rnd);
                    if (addition.isEmpty()) {
                      break;
                    }
                    newNetwork = addition.applyTo(newNetwork);
                  }
                  if (!avoidDeadGates || NetworkUtils.deadComparator().compare(newNetwork, n) <= 0) {
                    return newNetwork;
                  }
                } catch (NetworkStructureException | TypeException e) {
                  // ignore
                }
                newWires.remove(newWireBefore);
                newWires.remove(newWireAfter);
                nOfAttempts = nOfAttempts + 1;
              }
            }
          }
        }
      }
    }
    return n;
  }
}
