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

import io.github.ericmedvet.jgea.core.operator.Crossover;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.random.RandomGenerator;

public class NetworkCrossover implements Crossover<Network> {
  private final int maxNOfGates;
  private final double subnetSizeRate;
  private final int maxNOfAttempts;
  private final boolean avoidDeadGates;

  public NetworkCrossover(int maxNOfGates, double subnetSizeRate, int maxNOfAttempts, boolean avoidDeadGates) {
    this.maxNOfGates = maxNOfGates;
    this.subnetSizeRate = subnetSizeRate;
    this.maxNOfAttempts = maxNOfAttempts;
    this.avoidDeadGates = avoidDeadGates;
  }

  @Override
  public Network recombine(Network n1, Network n2, RandomGenerator rnd) {
    for (int nOfAttempts = 0; nOfAttempts < maxNOfAttempts; nOfAttempts = nOfAttempts + 1) {
      try {
        int innerSize1 = (int) n1.gates()
            .values()
            .stream()
            .filter(g -> !(g instanceof Gate.InputGate) && !(g instanceof Gate.OutputGate))
            .count();
        int innerSize2 = (int) n2.gates()
            .values()
            .stream()
            .filter(g -> !(g instanceof Gate.InputGate) && !(g instanceof Gate.OutputGate))
            .count();
        Network hn1 = NetworkUtils.randomHoledNetwork(n1, rnd, (int) Math.max(1, innerSize1 * subnetSizeRate));
        Network sn2 = NetworkUtils.randomSubnetwork(
            n2,
            rnd,
            Math.min((int) Math.max(1, innerSize2 * subnetSizeRate), maxNOfGates - hn1.gates().size())
        );
        int maxGI = hn1.gates().keySet().stream().mapToInt(i -> i).max().orElse(0);
        Map<Integer, Gate> newGates = new HashMap<>(hn1.gates());
        Set<Wire> newWires = new LinkedHashSet<>(hn1.wires());
        sn2.gates().forEach((gi, g) -> newGates.put(gi + maxGI + 1, g));
        sn2.wires()
            .forEach(
                w -> newWires.add(
                    Wire.of(
                        w.src().gateIndex() + maxGI + 1,
                        w.src().portIndex(),
                        w.dst().gateIndex() + maxGI + 1,
                        w.dst().portIndex()
                    )
                )
            );
        Network newN = new Network(newGates, newWires);
        // wire everything
        while (true) {
          Network.Addition addition = NetworkUtils.wire(newN, false, rnd);
          if (addition.isEmpty()) {
            break;
          }
          newN = addition.applyTo(newN);
        }
        if (!avoidDeadGates || (NetworkUtils.deadComparator().compare(newN, n1) <= 0 && NetworkUtils.deadComparator()
            .compare(newN, n2) <= 0)) {
          return newN;
        }
      } catch (NetworkStructureException | TypeException e) {
        // ignore
      }
    }
    return n1;
  }
}
