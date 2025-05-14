/*-
 * ========================LICENSE_START=================================
 * jgea-problem
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
package io.github.ericmedvet.jgea.problem.extraction;

import io.github.ericmedvet.jgea.core.problem.SimpleEBMOProblem;
import io.github.ericmedvet.jgea.core.representation.graph.finiteautomata.Extractor;
import io.github.ericmedvet.jgea.core.util.IntRange;
import io.github.ericmedvet.jgea.core.util.Misc;
import io.github.ericmedvet.jnb.datastructure.TriFunction;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

public interface ExtractionProblem<A> extends SimpleEBMOProblem<Extractor<A>, List<A>, Set<IntRange>, ExtractionProblem.Outcome, Double> {
  record Outcome(int length, Set<IntRange> desired, Set<IntRange> extracted) {}

  enum Metric {
    ONE_MINUS_PREC, ONE_MINUS_REC, ONE_MINUS_FM, SYMBOL_FNR, SYMBOL_FPR, SYMBOL_ERROR, SYMBOL_WEIGHTED_ERROR;

    @Override
    public String toString() {
      return name().toLowerCase();
    }
  }

  List<Metric> metrics();

  @Override
  default SequencedMap<String, Objective<List<Outcome>, Double>> aggregateObjectives() {
    return metrics().stream()
        .collect(
            Misc.toSequencedMap(
                Enum::toString,
                m -> new Objective<>(
                    outcomes -> aggregateFunction().apply(outcomes).get(m.toString()), // should never be called
                    Double::compareTo
                )
            )
        );
  }

  @Override
  default TriFunction<List<A>, Set<IntRange>, Set<IntRange>, Outcome> errorFunction() {
    return (as, desiredExtractions, extractions) -> new Outcome(as.size(), desiredExtractions, extractions);
  }

  @Override
  default Function<List<Outcome>, SequencedMap<String, Double>> aggregateFunction() {
    return results -> {
      SequencedMap<String, List<Double>> raw = new LinkedHashMap<>();
      Set<Metric> metrics = new LinkedHashSet<>(metrics());
      metrics.forEach(m -> raw.put(m.toString(), new ArrayList<>()));
      results.forEach(outcome -> computeForResult(outcome, metrics).forEach((k, v) -> raw.get(k).add(v)));
      return Misc.sequencedTransformValues(raw, vs -> vs.stream().mapToDouble(v -> v).average().orElseThrow());
    };
  }

  @Override
  default BiFunction<Extractor<A>, List<A>, Set<IntRange>> predictFunction() {
    return Extractor::extractNonOverlapping;
  }

  private static SequencedMap<String, Double> computeForResult(Outcome outcome, Set<Metric> metrics) {
    Map<Metric, Double> values = new EnumMap<>(Metric.class);
    if (metrics.contains(Metric.ONE_MINUS_FM) || metrics.contains(Metric.ONE_MINUS_PREC) || metrics.contains(
        Metric.ONE_MINUS_REC
    )) {
      // precision and recall
      Set<IntRange> correctExtractions = new LinkedHashSet<>(outcome.extracted);
      correctExtractions.retainAll(outcome.desired);
      double recall = (double) correctExtractions.size() / (double) outcome.desired.size();
      double precision = (double) correctExtractions.size() / (double) outcome.extracted.size();
      double fMeasure = 2d * precision * recall / (precision + recall);
      values.put(Metric.ONE_MINUS_PREC, 1 - precision);
      values.put(Metric.ONE_MINUS_REC, 1 - recall);
      values.put(Metric.ONE_MINUS_FM, 1 - fMeasure);
    }
    if (metrics.contains(Metric.SYMBOL_ERROR) || metrics.contains(Metric.SYMBOL_FNR) || metrics.contains(
        Metric.SYMBOL_FPR
    ) || metrics.contains(Metric.SYMBOL_WEIGHTED_ERROR)) {
      BitSet extractionMask = buildMask(outcome.extracted, outcome.length);
      BitSet desiredExtractionMask = buildMask(outcome.desired, outcome.length);
      int extractedSymbols = extractionMask.cardinality();
      extractionMask.and(desiredExtractionMask);
      double truePositiveSymbols = extractionMask.cardinality();
      double falseNegativeSymbols = desiredExtractionMask.cardinality() - truePositiveSymbols;
      double falsePositiveSymbols = extractedSymbols - truePositiveSymbols;
      double trueNegativeChars = desiredExtractionMask
          .length() - falsePositiveSymbols - truePositiveSymbols - falseNegativeSymbols;
      values.put(Metric.SYMBOL_FPR, falsePositiveSymbols / (trueNegativeChars + falsePositiveSymbols));
      values.put(Metric.SYMBOL_FNR, falseNegativeSymbols / (truePositiveSymbols + falseNegativeSymbols));
      values.put(
          Metric.SYMBOL_ERROR,
          (falsePositiveSymbols + falseNegativeSymbols) / (double) outcome.length
      );
      values.put(
          Metric.SYMBOL_WEIGHTED_ERROR,
          (falsePositiveSymbols / (trueNegativeChars + falsePositiveSymbols) + falseNegativeSymbols / (truePositiveSymbols + falseNegativeSymbols)) / 2d
      );
    }
    return metrics.stream()
        .collect(
            Misc.toSequencedMap(
                Enum::toString,
                values::get
            )
        );
  }

  private static BitSet buildMask(Set<IntRange> extractions, int size) {
    BitSet bitSet = new BitSet(size);
    extractions.forEach(r -> bitSet.set(r.min(), r.max()));
    return bitSet;
  }

}
