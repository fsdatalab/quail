//! Structs written with a `"type"` key naming their Python class.

/// Define a struct that reads and writes `{"type": <tag>, ...fields}`.
///
/// serde's own struct tag cannot be combined with
/// `deny_unknown_fields`, which then refuses the tag key. The struct
/// instead goes through a one-variant tagged enum: reading refuses
/// another tag, a missing field, and an unknown field. Further literals after the tag are older tags that are
/// also accepted.
#[macro_export]
macro_rules! tagged_struct {
    (
        $(#[$doc:meta])*
        $name:ident($body:ident, $wrapper:ident) = $tag:literal $(| $old:literal)* {
            $($(#[$field_doc:meta])* $field:ident: $field_type:ty $(= $absent:ident)?),* $(,)?
        }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            $($(#[$field_doc])* pub $field: $field_type,)*
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct $body {
            $($(#[serde($absent)])? $field: $field_type,)*
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(tag = "type")]
        enum $wrapper {
            #[serde(rename = $tag $(, alias = $old)*)]
            Value($body),
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serde::Serialize::serialize(&$wrapper::Value($body {
                    $($field: self.$field.clone(),)*
                }), serializer)

            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let $wrapper::Value(body) = <$wrapper as serde::Deserialize>::deserialize(deserializer)?;
                Ok($name {
                    $($field: body.$field,)*
                })
            }
        }
    };
}
