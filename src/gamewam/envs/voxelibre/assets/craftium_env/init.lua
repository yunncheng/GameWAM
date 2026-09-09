local task_kind = assert(minetest.settings:get("gamewam_task_kind"), "missing gamewam_task_kind")
local target_name = assert(minetest.settings:get("gamewam_target"), "missing gamewam_target")
local structure = assert(minetest.settings:get("gamewam_structure"), "missing gamewam_structure")
local inventory_string = assert(minetest.settings:get("gamewam_inventory"), "missing gamewam_inventory")
local selected_hotbar = tonumber(minetest.settings:get("gamewam_selected_hotbar")) or 1
assert(selected_hotbar == 1, "formal suite requires first hotbar slot")
local target_x = tonumber(minetest.settings:get("gamewam_target_x")) or 0
local target_z = tonumber(minetest.settings:get("gamewam_target_z")) or 5
local view_offset = tonumber(minetest.settings:get("gamewam_view_yaw_offset_deg")) or 0

local floor_y = 20
local target_pos = {x = math.floor(target_x + 0.5), y = floor_y + 1, z = math.floor(target_z + 0.5)}
local target_object = nil
local completed = false

local function assert_definition()
	if task_kind == "kill_entity" then
		assert(minetest.registered_entities[target_name], "unknown target entity: " .. target_name)
	else
		assert(minetest.registered_nodes[target_name], "unknown target node: " .. target_name)
	end
	for item in string.gmatch(inventory_string, "[^|]+") do
		local item_name = ItemStack(item):get_name()
		assert(minetest.registered_items[item_name], "unknown inventory item: " .. item_name)
	end
	assert(type(set_reward_once) == "function", "Craftium set_reward_once API unavailable")
	assert(type(set_termination) == "function", "Craftium set_termination API unavailable")
end

local function finish_success()
	if completed then return end
	completed = true
	set_reward_once(1.0, 0.0)
	set_termination()
end

local function prepare_map()
	local minp = {x = -12, y = floor_y - 3, z = -12}
	local maxp = {x = 12, y = floor_y + 10, z = 12}
	local vm = VoxelManip()
	local emerged_min, emerged_max = vm:read_from_map(minp, maxp)
	local area = VoxelArea:new({MinEdge=emerged_min, MaxEdge=emerged_max})
	local data = vm:get_data()
	local air = minetest.get_content_id("air")
	local dirt = minetest.get_content_id("mcl_core:dirt")
	local grass = minetest.get_content_id("mcl_core:dirt_with_grass")
	for x = minp.x, maxp.x do
		for z = minp.z, maxp.z do
			for y = minp.y, maxp.y do
				local content = air
				if y < floor_y then content = dirt end
				if y == floor_y then content = grass end
				data[area:index(x, y, z)] = content
			end
		end
	end
	vm:set_data(data)
	vm:write_to_map(true)
	for offset = -12, 12 do
		for y = floor_y + 1, floor_y + 4 do
			minetest.set_node({x=-12, y=y, z=offset}, {name="mcl_core:barrier"})
			minetest.set_node({x=12, y=y, z=offset}, {name="mcl_core:barrier"})
			minetest.set_node({x=offset, y=y, z=-12}, {name="mcl_core:barrier"})
			minetest.set_node({x=offset, y=y, z=12}, {name="mcl_core:barrier"})
		end
	end
	for _, object in ipairs(minetest.get_objects_inside_radius({x=0, y=floor_y+1, z=0}, 16)) do
		if not object:is_player() then object:remove() end
	end
end

local function build_structure()
	if structure == "tree" then
		for y = floor_y + 1, floor_y + 4 do
			minetest.set_node({x=target_pos.x, y=y, z=target_pos.z}, {name=target_name})
		end
		for x = target_pos.x - 2, target_pos.x + 2 do
			for z = target_pos.z - 2, target_pos.z + 2 do
				for y = floor_y + 4, floor_y + 6 do
					if math.abs(x-target_pos.x) + math.abs(z-target_pos.z) < 4 then
						minetest.set_node({x=x, y=y, z=z}, {name="mcl_core:leaves"})
					end
				end
			end
		end
		minetest.set_node({x=target_pos.x, y=floor_y+4, z=target_pos.z}, {name=target_name})
	elseif structure == "ore_outcrop" then
		for x = target_pos.x - 1, target_pos.x + 1 do
			for y = floor_y + 1, floor_y + 2 do
				minetest.set_node({x=x, y=y, z=target_pos.z}, {name=target_name})
			end
		end
	elseif structure == "placement_pedestal" then
		minetest.set_node(target_pos, {name="mcl_core:stone"})
	elseif structure == "single" then
		target_object = minetest.add_entity(target_pos, target_name)
		assert(target_object, "failed to spawn target entity: " .. target_name)
		local entity = target_object:get_luaentity()
		assert(entity, "spawned target has no Lua entity: " .. target_name)
		entity.persistent = true
		local original_on_die = entity.on_die
		entity.on_die = function(self, pos, cause)
			local puncher = cause and cause.type == "punch" and cause.puncher
			if puncher and puncher:is_player() then finish_success() end
			if original_on_die then return original_on_die(self, pos, cause) end
		end
		if entity and target_name == "mobs_mc:zombie" then
			entity.ignited_by_sunlight = false
			entity.sunlight_damage = 0
		end
	else
		error("unknown gamewam_structure: " .. structure)
	end
end

minetest.register_on_mods_loaded(assert_definition)

minetest.register_on_joinplayer(function(player)
	minetest.set_timeofday(0.5)
	if mcl_weather and mcl_weather.change_weather then
		pcall(mcl_weather.change_weather, "none", nil, "gamewam")
	end
	prepare_map()
	build_structure()

	local inv = player:get_inventory()
	inv:set_list("main", {})
	local slot = selected_hotbar
	for item in string.gmatch(inventory_string, "[^|]+") do
		inv:set_stack("main", slot, ItemStack(item))
		slot = slot + 1
	end
	assert(player:get_wield_index() == selected_hotbar,
		"fresh VoxeLibre player did not start on the formal hotbar slot")
	player:set_pos({x=0, y=floor_y+0.5, z=0})
	player:set_hp(20)
	player:hud_set_flags({crosshair=true, basic_debug=false, chat=false})

	-- Luanti first-person eye height is 1.625 nodes. Aim node tasks at
	-- the target node center and entity tasks at torso height.
	local eye = vector.add(player:get_pos(), {x=0, y=1.625, z=0})
	local aim_height = task_kind == "kill_entity" and 0.8 or 0.0
	local aim = vector.add(target_pos, {x=0, y=aim_height, z=0})
	local direction = vector.direction(eye, aim)
	player:set_look_horizontal(minetest.dir_to_yaw(direction) + math.rad(view_offset))
	player:set_look_vertical(-math.asin(direction.y))
end)

minetest.register_on_dignode(function(_pos, oldnode, digger)
	if not completed and task_kind == "dig_node" and digger and digger:is_player()
		and oldnode.name == target_name then
		finish_success()
	end
end)

minetest.register_on_placenode(function(_pos, newnode, placer)
	if not completed and task_kind == "place_node" and placer and placer:is_player()
		and newnode.name == target_name then
		finish_success()
	end
end)

minetest.register_on_dieplayer(function()
	if not completed then set_termination() end
end)

minetest.register_globalstep(function()
	minetest.set_timeofday(0.5)
end)
