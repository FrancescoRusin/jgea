/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.malelab.jgea.representation.grammar.cfggp;

import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.problem.symbolicregression.Element;
import it.units.malelab.jgea.problem.symbolicregression.SymbolicRegressionGrammar;
import it.units.malelab.jgea.representation.grammar.Grammar;
import it.units.malelab.jgea.representation.tree.Tree;

import java.util.List;
import java.util.Random;
import java.util.random.RandomGenerator;

/**
 * @author eric
 */
public class GrammarBasedSubtreeMutation<T> implements Mutation<Tree<T>> {

  private final int maxDepth;
  private final GrowGrammarTreeFactory<T> factory;

  public GrammarBasedSubtreeMutation(int maxDepth, Grammar<T> grammar) {
    this.maxDepth = maxDepth;
    factory = new GrowGrammarTreeFactory<>(0, grammar);
  }

  public static void main(String[] args) {
    SymbolicRegressionGrammar g = new SymbolicRegressionGrammar(
        List.of(Element.Operator.ADDITION, Element.Operator.MULTIPLICATION, Element.Operator.COS),
        List.of("x", "y"),
        List.of(0.1, 1d)
    );
    System.out.println(g);
    GrowGrammarTreeFactory<String> factory = new GrowGrammarTreeFactory<>(4, g);
    RandomGenerator r = new Random(2);
    Tree<String> t = factory.build(1, r).get(0);
    //t.prettyPrint(System.out);
    System.out.println(t);
    t.topSubtrees().forEach(System.out::println);
  }

  @Override
  public Tree<T> mutate(Tree<T> parent, RandomGenerator random) {
    Tree<T> child = Tree.copyOf(parent);
    List<Tree<T>> nonTerminalTrees = Misc.shuffle(child.topSubtrees(), random);
    boolean done = false;
    for (Tree<T> toReplaceSubTree : nonTerminalTrees) {
      // TODO should select a depth randomly such that the resulting child is <= maxDepth
      Tree<T> newSubTree = factory.build(random, toReplaceSubTree.content(), toReplaceSubTree.height());
      if (newSubTree != null) {
        toReplaceSubTree.clearChildren();
        newSubTree.childStream().forEach(toReplaceSubTree::addChild);
        done = true;
        break;
      }
    }
    if (!done) {
      return null;
    }
    return child;
  }
}
