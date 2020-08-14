/*
 * Copyright (C) 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package it.units.malelab.jgea.representation.graph.numeric.operatorgraph;

import com.google.common.graph.ImmutableValueGraph;
import com.google.common.graph.MutableValueGraph;
import com.google.common.graph.ValueGraph;
import com.google.common.graph.ValueGraphBuilder;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.representation.graph.numeric.Constant;
import it.units.malelab.jgea.representation.graph.numeric.Input;
import it.units.malelab.jgea.representation.graph.numeric.Node;
import it.units.malelab.jgea.representation.graph.numeric.Output;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * @author eric
 * @created 2020/08/14
 * @project jgea
 */
public class ShallowFactory implements IndependentFactory<ValueGraph<Node, OperatorGraph.Edge>> {
  private final int nInputs;
  private final int nOutputs;
  private final Constant[] constants;

  public ShallowFactory(int nInputs, int nOutputs, double... constants) {
    this.nInputs = nInputs;
    this.nOutputs = nOutputs;
    this.constants = new Constant[constants.length];
    for (int i = 0; i < constants.length; i++) {
      this.constants[i] = new Constant(i, constants[i]);
    }
  }

  @Override
  public ValueGraph<Node, OperatorGraph.Edge> build(Random random) {
    MutableValueGraph<Node, OperatorGraph.Edge> g = ValueGraphBuilder.directed().allowsSelfLoops(false).build();
    Input[] inputs = new Input[nInputs];
    Output[] outputs = new Output[nOutputs];
    for (int i = 0; i < nInputs; i++) {
      inputs[i] = new Input(i);
      g.addNode(inputs[i]);
    }
    for (int o = 0; o < nOutputs; o++) {
      outputs[o] = new Output(o);
      g.addNode(outputs[o]);
    }
    for (int i = 0; i < nInputs; i++) {
      inputs[i] = new Input(i);
      g.addNode(inputs[i]);
    }
    for (int c = 0; c < constants.length; c++) {
      g.addNode(constants[c]);
    }
    for (int o = 0; o < nOutputs; o++) {
      if (random.nextBoolean()) {
        g.putEdgeValue(inputs[random.nextInt(inputs.length)], outputs[o], OperatorGraph.EDGE);
      } else {
        g.putEdgeValue(constants[random.nextInt(constants.length)], outputs[o], OperatorGraph.EDGE);
      }
    }
    return ImmutableValueGraph.copyOf(g);
  }
}
