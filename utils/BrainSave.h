#pragma once

#include "../brain/Brain.h"

#include <filesystem>

class Brain;

namespace BrainSave {

std::filesystem::path save(
	const Brain& brain,
	const std::filesystem::path& dir
);

Brain load(
	const std::filesystem::path& file
);

int nextBrainNumber(
	const std::filesystem::path& dir
);

}
