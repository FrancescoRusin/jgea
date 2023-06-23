/*
 * Copyright 2023 eric
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

package io.github.ericmedvet.jgea.problem.grid;

import io.github.ericmedvet.jgea.core.problem.ComparableQualityBasedProblem;
import io.github.ericmedvet.jsdynsym.grid.Grid;
import io.github.ericmedvet.jsdynsym.grid.GridUtils;

import java.util.function.Function;

public class CharShapeApproximation implements ComparableQualityBasedProblem<Grid<Character>, Double> {

  private final Grid<Character> target;
  private final boolean translation;

  public CharShapeApproximation(Grid<Character> target, boolean translation) {
    this.target = target;
    this.translation = translation;
  }


  private static Grid.Key center(Grid<?> grid) {
    return new Grid.Key(
        (int) grid.entries()
            .stream()
            .filter(e -> e.value() != null)
            .mapToInt(e -> e.key().x())
            .average()
            .orElse(0d),
        (int) grid.entries()
            .stream()
            .filter(e -> e.value() != null)
            .mapToInt(e -> e.key().y())
            .average()
            .orElse(0d)
    );
  }

  @Override
  public Function<Grid<Character>, Double> qualityFunction() {
    Grid.Key targetCenter = center(target);
    return grid -> {
      Grid<Character> thisGrid = grid;
      Grid<Character> targetGrid = target;
      if (translation) {
        //possibly translate
        Grid.Key thisCenter = center(thisGrid);
        Grid.Key newCenter = new Grid.Key(
            Math.max(thisCenter.x(), targetCenter.x()),
            Math.max(thisCenter.y(), targetCenter.y())
        );
        thisGrid = GridUtils.translate(
            thisGrid,
            new Grid.Key(newCenter.x() - thisCenter.x(), newCenter.y() - thisCenter.y())
        );
        targetGrid = GridUtils.translate(
            targetGrid,
            new Grid.Key(newCenter.x() - targetCenter.x(), newCenter.y() - targetCenter.y())
        );
      }
      final Grid<Character> finalThisGrid = thisGrid;
      return targetGrid.entries().stream()
          .filter(e -> finalThisGrid.isValid(e.key()))
          .mapToDouble(e -> {
            if (e.value() == null && finalThisGrid.get(e.key()) == null) {
              return 1d;
            }
            if (e.value() == null) {
              return 0d;
            }
            if (e.value().equals(finalThisGrid.get(e.key()))) {
              return 1d;
            }
            return 0d;
          })
          .average()
          .orElse(0d);
    };
  }
}