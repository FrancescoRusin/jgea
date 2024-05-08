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
package io.github.ericmedvet.jgea.experimenter.listener.decoupled;

import io.github.ericmedvet.jnb.datastructure.Pair;
import java.time.LocalDateTime;

/**
 * @author "Eric Medvet" on 2023/11/04 for jgea
 */
public class DirectSinkSource<K, V> extends AbstractAutoPurgingSource<K, V> implements Sink<K, V> {

  @Override
  public void close() {
    super.close();
  }

  @Override
  public void push(LocalDateTime t, K k, V v) {
    synchronized (map) {
      map.put(new Pair<>(t, k), v);
    }
  }
}
