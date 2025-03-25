NVCC = $(shell which nvcc)
NVCC_FLAGS := -std=c++14 -g -Xcompiler -O0 -Xcicc -O2 -arch=sm_87
NVCC_FLAGS += --expt-relaxed-constexpr
NVCC_FLAGS += -arch=sm_87

OUTPUT_DIR = output
OUTPUT = $(OUTPUT_DIR)/MP
# PTX = MP_PTX

SRCS := message_passing.cu

VAR_SCOPES_X = X_CTA X_DEV
VAR_SCOPES_Y = Y_CTA Y_DEV Y_SYS
# VAR_SCOPES_X = X_DEV
# VAR_SCOPES_Y = Y_DEV

PRODUCER_FENCES = PRODUCER_FENCE_SC PRODUCER_FENCE_ACQ_REL PRODUCER_NO_FENCE
PRODUCER_FENCE_SCOPES = PRODUCER_FENCE_SCOPE_SYS PRODUCER_FENCE_SCOPE_CTA PRODUCER_FENCE_SCOPE_DEV 

CONSUMER_FENCES = CONSUMER_FENCE_SC CONSUMER_FENCE_ACQ_REL CONSUMER_NO_FENCE
CONSUMER_FENCE_SCOPES = CONSUMER_FENCE_SCOPE_SYS CONSUMER_FENCE_SCOPE_CTA CONSUMER_FENCE_SCOPE_DEV 

ORDERS = RLX_RLX REL_ACQ RLX_ACQ REL_RLX


all: $(foreach x_scope,$(VAR_SCOPES_X),$(foreach y_scope,$(VAR_SCOPES_Y),$(foreach producer_fence,$(PRODUCER_FENCES),$(foreach producer_scope,$(PRODUCER_FENCE_SCOPES),$(foreach consumer_fence,$(CONSUMER_FENCES),$(foreach consumer_scope,$(CONSUMER_FENCE_SCOPES),$(foreach order,$(ORDERS),$(OUTPUT)-$(x_scope)-$(y_scope)-$(producer_fence)-$(producer_scope)-$(consumer_fence)-$(consumer_scope)-$(order).out)))))))


define make_target
$(OUTPUT)-$(1)-$(2)-$(3)-$(4)-$(5)-$(6)-$(7).out: $(SOURCES)
	mkdir -p $(OUTPUT_DIR)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -D$(4) -D$(5) -D$(6) -D$(7) -o $$@ $$(SRCS)
endef


$(foreach x_scope,$(VAR_SCOPES_X),$(foreach y_scope,$(VAR_SCOPES_Y),$(foreach producer_fence,$(PRODUCER_FENCES),$(foreach producer_scope,$(PRODUCER_FENCE_SCOPES),$(foreach consumer_fence,$(CONSUMER_FENCES),$(foreach consumer_scope,$(CONSUMER_FENCE_SCOPES),$(foreach order,$(ORDERS),$(eval $(call make_target,$(x_scope),$(y_scope),$(producer_fence),$(producer_scope),$(consumer_fence),$(consumer_scope),$(order))))))))))

clean:
	rm -f $(OUTPUT)*.out