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

package io.github.ericmedvet.jgea.core.representation.sequence.bit;

import io.github.ericmedvet.jgea.core.IndependentFactory;
import java.util.random.RandomGenerator;

public class BitStringFactory implements IndependentFactory<BitString> {

    private final int size;

    public BitStringFactory(int size) {
        this.size = size;
    }

    @Override
    public BitString build(RandomGenerator random) {
        BitString bitString = new BitString(size);
        for (int i = 0; i < bitString.size(); i++) {
            bitString.bits()[i] = random.nextBoolean();
        }
        return bitString;
    }
}
