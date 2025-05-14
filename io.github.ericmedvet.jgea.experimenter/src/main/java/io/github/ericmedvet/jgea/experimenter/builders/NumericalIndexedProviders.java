/*-
 * ========================LICENSE_START=================================
 * jgea-experimenter
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
package io.github.ericmedvet.jgea.experimenter.builders;

import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import io.github.ericmedvet.jgea.problem.regression.NumericalDataset;
import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

@Discoverable(prefixTemplate = "ea.provider.numerical|num")
public class NumericalIndexedProviders {
  private NumericalIndexedProviders() {
  }

  @Cacheable
  public static NumericalDataset empty(
      @Param("xVars") List<String> xVarNames,
      @Param("yVars") List<String> yVarNames
  ) {
    return NumericalDataset.from(xVarNames, yVarNames, IndexedProvider.from(List.of()));
  }

  @Cacheable
  public static NumericalDataset fromFile(
      @Param("filePath") String filePath,
      @Param(value = "xVarNamePattern", dS = "x.*") String xVarNamePattern,
      @Param(value = "yVarNamePattern", dS = "y.*") String yVarNamePattern,
      @Param(value = "limit", dI = Integer.MAX_VALUE) int limit
  ) {
    try (InputStream is = new FileInputStream(filePath)) {
      return NumericalDataset.fromCSV(xVarNamePattern, yVarNamePattern, is, limit);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Cacheable
  public static NumericalDataset fromBundled(
      @Param("name") String name,
      @Param(value = "xScaling", dS = "none") NumericalDataset.Scaling xScaling,
      @Param(value = "yScaling", dS = "none") NumericalDataset.Scaling yScaling,
      @Param(value = "limit", dI = Integer.MAX_VALUE) int limit
  ) {
    try {
      NumericalDataset dataset = switch (name) {
        case "concrete" -> NumericalDataset.fromResourceCSV(".*", "strength", name, limit);
        case "wine" -> NumericalDataset.fromResourceCSV(".*", "quality", name, limit);
        case "energy-efficiency" -> NumericalDataset.fromResourceCSV("x[0-9]+", "y1", name, limit);
        case "xor" -> NumericalDataset.fromResourceCSV(".*", "y", name, limit);
        default -> throw new IllegalArgumentException("Unknown bundled dataset: %s".formatted(name));
      };
      return dataset.xScaled(xScaling).yScaled(yScaling).prepared();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Cacheable
  public static NumericalDataset scaled(
      @Param("of") NumericalDataset dataset,
      @Param(value = "xScaling", dS = "none") NumericalDataset.Scaling xScaling,
      @Param(value = "yScaling", dS = "none") NumericalDataset.Scaling yScaling
  ) {
    return dataset.xScaled(xScaling).yScaled(yScaling);
  }
}
