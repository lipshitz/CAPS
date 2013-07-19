#include <tr1/tuple>
#include <tuple>

int main() {
  std::tr1::tuple<int,char> foo (10,'x');
  auto bar = std::make_tuple(10,'x');
}
