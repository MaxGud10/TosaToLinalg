# TosaToLinalg

Учебный проект на **MLIR**, который реализует **progressive lowering** из диалекта **TOSA** в **Linalg-on-tensors**.

## Описание проекта

Во многих ML-компиляторах высокоуровневые операции нейронных сетей сначала представляются в предметно-ориентированном диалекте, а затем понижаются в более универсальное промежуточное представление. В этом проекте таким переходом является конвертация из **TOSA** в **Linalg**.

Цель проекта - показать, как операции уровня нейросетей:

- `tosa.add`
- `tosa.mul`
- `tosa.matmul`
- `tosa.clamp`

можно представить через:

- `tensor.empty`
- `linalg.fill`
- `linalg.generic`
- `arith.*`

То есть проект переводит TOSA-операции в описание вычислений через **пространство итераций**, **indexing maps** и **region body** внутри `linalg.generic`.

---

## Что реализовано

### 1. Настройка `Dialect Conversion`

В pass'е настраивается `ConversionTarget`, в котором:

- целевые TOSA-операции объявлены **illegal**;
- диалекты `linalg`, `tensor`, `arith`, `func` объявлены **legal**.

Для применения преобразования используется `applyFullConversion`, поэтому pass считается успешным только тогда, когда все объявленные нелегальные операции были заменены.

### 2. Lowering `tosa.add`

Операция `tosa.add` понижается в `linalg.generic`:

- создаётся выходной тензор через `tensor.empty`;
- строятся `indexing_maps`;
- все итераторы объявляются как `parallel`;
- внутри тела `linalg.generic` создаётся `arith.addf` или `arith.addi`;
- результат возвращается через `linalg.yield`.

### 3. Lowering `tosa.mul`

Операция `tosa.mul` также понижается в `linalg.generic`:

- создаётся выходной тензор;
- строится elementwise-итерационное пространство;
- внутри region body используется `arith.mulf` или `arith.muli`.

### 4. Lowering `tosa.matmul`

Операция `tosa.matmul` понижается в `linalg.generic` как batched matrix multiplication:

- создаётся выходной тензор через `tensor.empty`;
- затем он инициализируется нулями через `linalg.fill`;
- используется `linalg.generic` с четырьмя итераторами:
  - `parallel`
  - `parallel`
  - `parallel`
  - `reduction`
- внутри тела выполняется схема:
  - умножение двух элементов;
  - накопление в аккумулятор.

Таким образом, `tosa.matmul` переводится в явное описание вычисления через пространство итераций и reduction dimension.

### 5. Lowering `tosa.clamp`

Операция `tosa.clamp` используется как аналог простой ReLU-подобной нелинейности в тестовом графе:

- понижается в `linalg.generic`;
- внутри тела используются min/max операции из `arith`.

### 6. Работа с тензорами

Проект использует `tensor.empty` для создания выходных тензоров и корректно пробрасывает:

- shape;
- element type;
- dynamic dimensions.


## Сборка

Проект собирается как *out-of-tree MLIR projec*, поэтому требуется заранее собранный `llvm-project` с включённым `mlir`.

1. Сборка проекта
```bash
git clone https://github.com/MaxGud10/TosaToLinalg
mkdir build
cd build


cmake -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  ..

cmake --build . -j2
```

Запуск
```bash
./tosa-to-linalg-opt --pass-pipeline="builtin.module(lower-tosa-to-linalg)" ../test/add.mlir
```

### Что должно произойти ?
`add.mlir`

Вместо `tosa.add` должен появиться `linalg.generic`, внутри которого выполняется `arith.addf` или `arith.addi`

`mul.mlir`

Вместо `tosa.mul` должен появиться `linalg.generic`, внутри которого выполняется `arith.mulf` или `arith.muli`.

