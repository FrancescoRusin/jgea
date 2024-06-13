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
package io.github.ericmedvet.jgea.core.listener;

import java.util.ArrayList;
import java.util.List;
import java.util.function.*;

public interface AccumulatorFactory<E, O, K> extends ListenerFactory<E, K> {
  Accumulator<E, O> build(K k);

  @Override
  default AccumulatorFactory<E, O, K> conditional(Predicate<K> predicate) {
    AccumulatorFactory<E, O, K> thisAccumulatorFactory = this;
    return new AccumulatorFactory<>() {
      @Override
      public Accumulator<E, O> build(K k) {
        if (predicate.test(k)) {
          return thisAccumulatorFactory.build(k);
        }
        return Accumulator.nullAccumulator();
      }

      @Override
      public void shutdown() {
        thisAccumulatorFactory.shutdown();
      }
    };
  }

  static <E, O, K> AccumulatorFactory<E, List<O>, K> collector(Function<E, O> function) {
    return k -> Accumulator.collector(function);
  }

  static <E, O, K> AccumulatorFactory<E, O, K> last(BiFunction<E, K, O> function) {
    return k -> Accumulator.<E>last().then(e -> function.apply(e, k));
  }

  @Override
  default <X> AccumulatorFactory<X, O, K> on(Function<X, E> function) {
    AccumulatorFactory<E, O, K> thisAccumulatorFactory = this;
    return new AccumulatorFactory<>() {
      @Override
      public Accumulator<X, O> build(K k) {
        return thisAccumulatorFactory.build(k).on(function);
      }

      @Override
      public void shutdown() {
        thisAccumulatorFactory.shutdown();
      }
    };
  }

  default <Q> AccumulatorFactory<E, Q, K> then(Function<O, Q> function) {
    AccumulatorFactory<E, O, K> thisAccumulatorFactory = this;
    return new AccumulatorFactory<>() {
      @Override
      public Accumulator<E, Q> build(K k) {
        return thisAccumulatorFactory.build(k).then(function);
      }

      @Override
      public void shutdown() {
        thisAccumulatorFactory.shutdown();
      }
    };
  }

  default AccumulatorFactory<E, O, K> thenOnDone(BiConsumer<K, O> consumer) {
    AccumulatorFactory<E, O, K> thisFactory = this;
    return new AccumulatorFactory<>() {
      @Override
      public Accumulator<E, O> build(K k) {
        return thisFactory.build(k).thenOnDone(o -> consumer.accept(k, o));
      }

      @Override
      public void shutdown() {
        thisFactory.shutdown();
      }
    };
  }

  default AccumulatorFactory<E, O, K> thenOnShutdown(Consumer<List<O>> consumer) {
    AccumulatorFactory<E, O, K> thisAccumulatorFactory = this;
    List<O> os = new ArrayList<>();
    return new AccumulatorFactory<>() {
      @Override
      public Accumulator<E, O> build(K k) {
        Accumulator<E, O> accumulator = thisAccumulatorFactory.build(k);
        return new Accumulator<E, O>() {
          @Override
          public O get() {
            return accumulator.get();
          }

          @Override
          public void listen(E e) {
            accumulator.listen(e);
          }

          @Override
          public void done() {
            os.add(accumulator.get());
          }

          @Override
          public String toString() {
            return accumulator + "[thenOnShutDown:%s]".formatted(consumer);
          }
        };
      }

      @Override
      public void shutdown() {
        consumer.accept(os);
        thisAccumulatorFactory.shutdown();
      }
    };
  }
}
