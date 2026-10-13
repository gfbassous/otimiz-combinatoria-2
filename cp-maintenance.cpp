#include <stdlib.h>
#include <fstream>

#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>
#include <stdexcept>

#include "absl/strings/str_format.h"
#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

#include "DataFrame/DataFrame.h"

using namespace std;
using namespace operations_research;
using namespace sat;
using namespace hmdf;

vector<vector<string>> ReadCSV(string filename) {
  vector<vector<string>> rows;
      ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the file.\n";
        return rows;
    }

  string line;
  // Read the file line by line
  while (getline(file, line)) {
    stringstream ss(line);
    string cell;
    vector<string> row;

    // Split each line by commas
    while (getline(ss, cell, ',')) {
        row.push_back(cell);
    }
    rows.push_back(row);
}
  file.close();
  return rows;
}

int ScheduleMaintenanceSat() {
  string filename = "C:/work/otimiz-combinatoria-2/RTS-GMLC-master/RTS_Data/SourceData/gen.csv";
  auto file = ReadCSV(filename);
  
  return 1;
}

int main() {
  ScheduleMaintenanceSat();
  return EXIT_SUCCESS;
}