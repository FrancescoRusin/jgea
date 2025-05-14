/*-
 * ========================LICENSE_START=================================
 * jgea-problem
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

package io.github.ericmedvet.jgea.problem.extraction.string;

import io.github.ericmedvet.jgea.core.problem.TargetEBProblem;
import io.github.ericmedvet.jgea.core.representation.graph.finiteautomata.Extractor;
import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import io.github.ericmedvet.jgea.core.util.IntRange;
import io.github.ericmedvet.jgea.problem.extraction.ExtractionProblem;
import java.util.*;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public record RegexExtractionProblem(
    List<Metric> metrics,
    Function<? super List<Character>, ? extends Set<IntRange>> target,
    IndexedProvider<List<Character>> inputProvider,
    IndexedProvider<List<Character>> validationInputProvider
) implements ExtractionProblem<Character>, TargetEBProblem<Extractor<Character>, List<Character>, Set<IntRange>, ExtractionProblem.Outcome, SequencedMap<String, Double>> {

  private final static List<String> DEFAULT_REGEXES = List.of("000000", "111(00)?+(11)++", "(110110)++");

  public RegexExtractionProblem(
      List<Metric> metrics,
      String regex,
      IndexedProvider<String> inputProvider,
      IndexedProvider<String> validationInputProvider
  ) {
    this(
        metrics,
        cs -> new RegexBasedExtractor(regex).extract(cs),
        inputProvider.then(text -> text.chars().mapToObj(c -> (char) c).toList()),
        validationInputProvider.then(text -> text.chars().mapToObj(c -> (char) c).toList())
    );
  }

  public RegexExtractionProblem(
      List<Metric> metrics,
      int symbols,
      int size,
      long seed
  ) {
    this(
        metrics,
        DEFAULT_REGEXES.stream().map("(%s)"::formatted).collect(Collectors.joining("|")),
        IndexedProvider.from(
            List.of(
                buildText(
                    size,
                    DEFAULT_REGEXES,
                    "0123456789".substring(0, Math.min(symbols, 10)),
                    100,
                    new Random(seed)
                )
            )
        ),
        IndexedProvider.from(
            List.of(
                buildText(
                    size,
                    DEFAULT_REGEXES,
                    "0123456789".substring(0, Math.min(symbols, 10)),
                    100,
                    new Random(seed + 1)
                )
            )
        )
    );
  }

  private static String buildText(
      int minExtractionsPerRegex,
      List<String> regexes,
      String alphabet,
      int chunkSize,
      Random random
  ) {
    StringBuilder sb = new StringBuilder();
    while (true) {
      int initialLength = sb.length();
      while (sb.length() < initialLength + chunkSize) {
        sb.append(alphabet.charAt(random.nextInt(alphabet.length())));
      }
      int okPattern = 0;
      for (String regex : regexes) {
        int found = 0;
        Matcher matcher = Pattern.compile(regex).matcher(sb.toString());
        int s = 0;
        while (matcher.find(s)) {
          found = found + 1;
          s = matcher.end();
        }
        if (found > minExtractionsPerRegex) {
          okPattern = okPattern + 1;
        }
      }
      if (okPattern == regexes.size()) {
        return sb.toString();
      }
    }
  }
}
