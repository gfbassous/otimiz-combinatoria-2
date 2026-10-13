#include <iostream>
#include <ortools/linear_solver/linear_solver.h>

int _main() {
  using namespace operations_research;
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));
  MPVariable* x = solver->MakeNumVar(0, 10, "x");
  solver->MutableObjective()->SetCoefficient(x, 1);
  solver->MutableObjective()->SetMaximization();
  solver->Solve();
  std::cout << "x = " << x->solution_value() << "\n";  // expect 10
}